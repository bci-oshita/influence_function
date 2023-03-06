from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, Union

import numpy
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler, Sampler
from tqdm import tqdm

DataPoint = Union[numpy.ndarray, torch.Tensor, float, int]

random.seed(12345)


class InfluenceModel(nn.Module):
    def loss():
        raise NotImplementedError("loss()")


class SimpleDataset(TensorDataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor) -> None:
        assert len(data) == len(labels)
        self.data: torch.Tensor = data  # (B, D1, ..., Dn)
        self.labels: torch.Tensor = labels  # (B, )

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

    def __iter__(self) -> Iterator:
        for z in zip(self.data, self.labels):
            yield z

    def __len__(self) -> int:
        return len(self.data)


class SimpleSampler(Sampler):
    def __init__(self, dataset: TensorDataset, n_samples=-1, indices=[]) -> None:
        self.dataset = dataset
        self.n_samples = n_samples
        self.indices = (
            indices[:n_samples] if len(indices) > 0 else range(self.n_samples)
        )
        # assert len(self.indices) == n_samples

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        # return self.n_samples
        return len(self.indices)


def to_loader(
    dataset: TensorDataset | DataLoader,
    bsz: int = 1,
    do_shuffle: bool = False,
    n_samples: int = -1,
    seed: int = 12345,  # -1 : auto setting
    indices: list = [],  # use only do_shuffle = False
) -> DataLoader:
    if isinstance(dataset, DataLoader):
        return dataset

    n = n_samples if n_samples > 0 else (len(indices) if indices else len(dataset))

    # sampler の setup
    # # seed, generator
    if seed < 0:  # set randomly, i.e. no reproducing
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
    generator = torch.Generator().manual_seed(seed)

    # NOTE: DataLoader の sampler を使うと若干遅くなるかも
    sampler = (
        RandomSampler(dataset, num_samples=n * bsz, generator=generator)
        if do_shuffle
        else SimpleSampler(dataset, n * bsz, indices)
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bsz,
        shuffle=False,
        num_workers=2,
        sampler=sampler,
    )
    return dataloader


@dataclass
class Influence(object):
    test_idx: int
    values: list[torch.Tensor]
    helpful: list[torch.Tensor]  # index for self.values
    harmful: list[torch.Tensor]  # index for self.values
    helpful_train_indices: list[torch.Tensor]
    harmful_train_indices: list[torch.Tensor]
    values_index: dict[int, int]
    values_data: dict[int, int]  # train_idx -> influence value


def _filter_layername(p: str):
    return ".".join(p.split(".")[:-1])


class InfluenceCalculator(object):
    def __init__(
        self,
        damping: float = 0.1,
        scaling: float = 25.0,
        r: int = 5000,
        n_influence_samples: int = 1357,  # of train samples for calculating influence
        n_s_test_samples: int = 7,  # of train samples for calculating s_test
    ):
        self.model: InfluenceModel = None
        self.trainset: TensorDataset = None

        # hyper parameters
        self.damping = damping
        self.scaling = scaling

        # for the stabilitiy
        self.r = r
        self.n_influence_samples = n_influence_samples
        self.n_s_test_samples = n_s_test_samples

    def setup(
        self,
        model,
        trainset: TensorDataset,
        target_layers=[],
        device=torch.device("cpu"),
    ):
        self.model = model
        self.trainset = trainset
        self.target_layers = target_layers
        self.device: torch.device = device

        self.model.to(self.device)
        return self

    def _t(
        self, z: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return z[0].to(self.device), z[1].to(self.device)

    def calculate_simply(
        self,
        testset: TensorDataset | DataLoader,
        n_test_samples: int = -1,
        top_n: int = 500,
    ) -> dict:
        # NOTE: calculate s_test for each test point
        #       DO NOT do_shuffle=True, MUST do_shuffle=False
        testloader = to_loader(
            testset, bsz=1, do_shuffle=False, n_samples=n_test_samples
        )
        influence_data = {}
        for test_idx, z_test in enumerate(tqdm(testloader, desc="testloader")):
            x_test, t_test = self._t(z_test)
            influ = self.calculate_influences(test_idx, z_test)

            # make data
            test_id = str(test_idx)
            influence_data[test_id] = {}
            influence_data[test_id]["label"] = t_test.item()
            # FIXME: もとのデータセットのINDEXを保持する
            influence_data[test_id]["test_idx"] = test_idx
            influence_data[test_id]["time_calc_influence_s"] = -1
            influence_data[test_id]["influence"] = [
                _iv.cpu().numpy().tolist() for _iv in influ.values
            ]
            influence_data[test_id]["harmful"] = influ.harmful[:top_n]
            influence_data[test_id]["helpful"] = influ.helpful[:top_n]
        return influence_data

    def calculate_influences(self, test_idx: int, z_test: tuple) -> Influence:
        # NOTE: calculate s_test
        x_test, t_test = self._t(z_test)
        s_test = self._s_test(x_test, t_test)

        influence_values = []
        # NOTE: calculate influence function values
        #       DO NOT do_shuffle=True, MUST do_shuffle=False
        k = (
            self.n_influence_samples
            if self.n_influence_samples > 0
            else len(self.trainset)
        )
        sample_indices = random.sample(range(len(self.trainset)), k)
        d = to_loader(self.trainset, bsz=1, do_shuffle=False, indices=sample_indices)
        for z in tqdm(d, desc=f"calcuate influences [{test_idx=}]"):
            infval = self.calculate_influence_value(z, s_test)
            influence_values.append(infval.cpu().numpy())

        helpful = numpy.argsort(influence_values)
        harmful = helpful[::-1]

        # 元のtrainset index に変換
        helpful_train_indices = [sample_indices[idx] for idx in helpful]
        harmful_train_indices = [sample_indices[idx] for idx in harmful]

        values_index = {}
        values_data = {}
        for inner_idx, global_idx in enumerate(sample_indices):
            values_index[global_idx] = inner_idx
            values_data[global_idx] = influence_values[inner_idx]

        return Influence(
            test_idx=test_idx,
            values=influence_values,
            helpful=helpful.tolist(),
            harmful=harmful.tolist(),
            helpful_train_indices=helpful_train_indices,
            harmful_train_indices=harmful_train_indices,
            values_index=values_index,
            values_data=values_data,
        )

    def calculate_influence_value(self, z: tuple, s_test: list) -> float | torch.Tensor:
        # NOTE: return I_{up,loss}
        x, t = self._t(z)
        grad_z = self._nabla(x, t)

        # NOTE: $- s_{test} \dot \nabla_{\theta}L(z_{test}, \hat{\theta})$
        #       GPUを使うなら、torch.Tensor で算出した方が良いかも
        #       / n は、Hessian を計算するときにtraining points 上で平均をとるため
        n = len(self.trainset)
        influence_value = (
            -sum([torch.sum(Hinv * g).data for Hinv, g in zip(s_test, grad_z)]) / n
        )
        return influence_value

    def _s_test(self, x_test: DataPoint, t_test: DataPoint) -> list[torch.Tensor]:
        model: InfluenceModel = self.model

        # initialize
        try:
            v = self._nabla(x_test, t_test)
        except Exception as e:
            print(e, f"{self.target_layers=}")
            raise e
        H_tilder = v.copy()

        k = self.n_s_test_samples
        for r in tqdm(range(self.r), desc="recursion_depth"):
            # NOTE: H~ の近似計算を高速化するために、training point をサンプリングする
            # d = to_loader(self.trainset, bsz=1, do_shuffle=True, n_samples=k)
            d = to_loader(self.trainset, bsz=k, do_shuffle=True, n_samples=1)
            # assert len(d) == k
            for idz, z in enumerate(tqdm(d, desc=f"calculate s_test[{r=}]")):
                x, t = self._t(z)
                y = model(x)
                loss = model.loss(y, t)
                # サブモデル毎のパラメータ
                theta = self._params()
                Hv = self._hvp(loss, theta, H_tilder)

                H_tilder = [
                    _v + (1 - self.damping) * _H_tilder - _Hv.detach() / self.scaling
                    for _v, _H_tilder, _Hv in zip(v, H_tilder, Hv)
                ]
                # break

        H_tilder_avg = [H_ / k for H_ in H_tilder]
        return H_tilder_avg

    def _params(self) -> list[torch.Tensor]:
        # NOTE: target_layers が指定されてなければ、すべてを対象
        if not self.target_layers:
            return [p for p in self.model.parameters() if p.requires_grad]

        _target_params = []
        for nm, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            layer = _filter_layername(nm)
            if layer not in self.target_layers:
                continue
            _target_params.append(p)
        return _target_params

    def _nabla(self, x, t) -> list[torch.Tensor]:
        model = self.model
        model.eval()
        # model.zero_grad()
        y = model(x)
        loss = model.loss(y, t)
        theta = self._params()
        return list(grad(loss, theta, create_graph=True))

    def _hvp(self, loss, theta, vectors) -> tuple[torch.Tensor]:
        # Hessian Vecotr Products

        # # first derivative
        nablas = grad(loss, theta, retain_graph=True, create_graph=True)

        # calculate `nabla * v` for each parameter
        gv = torch.Tensor([0]).to(self.device)
        for g, v in zip(nablas, vectors):
            v = torch.clamp(v, -5e16, 5e16)
            gv += torch.sum(g * v)

        # # Second derivative
        Hv = grad(gv, theta, create_graph=True)

        return Hv
