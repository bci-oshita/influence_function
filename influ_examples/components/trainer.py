import joblib
import numpy
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Self
from .models.model import ModelClassify


class SimpleTrainer:
    def __init__(
        self,
        model: ModelClassify,
        optimizer: Optimizer,
        trainloader: DataLoader,
        testloader: DataLoader = None,
        device=torch.device("cpu"),
        n_validate_intervals: int = 10,
        n_intervals: int = 500,
    ) -> None:
        self.model: ModelClassify = model
        self.optimizer: Optimizer = optimizer
        self.trainloader: DataLoader = trainloader
        self.testloader: DataLoader = testloader
        self.device: torch.device = device

        self.n_validate_intervals = n_validate_intervals
        self.n_intervals = n_intervals

        self.model.to(self.device)

        # context
        self.losses: list = []

    def _t(self, z: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return z[0].to(self.device), z[1].to(self.device)

    def do_train(self, max_epoch=20, keyword="baseline_processing") -> Self:
        print(f"train.datasets: {len(self.trainloader.dataset.data)}")
        print(f"test.datasets: {len(self.testloader.dataset.data)}")

        # loop over the dataset multiple times
        for epoch in tqdm(range(max_epoch), desc="epoch"):
            self.running_loss = 0.0
            self.losses: list = []
            for bch_idx, z in enumerate(tqdm(self.trainloader, desc="batch"), 0):
                X, t = self._t(z)
                step = self.trainloader.batch_size * epoch + bch_idx

                # init grads
                self.optimizer.zero_grad()

                # train by step
                y = self.model(X)
                loss = self.model.loss(y, t)
                loss.backward()
                self.optimizer.step()

                # print statistics
                self.running_loss += loss.item()
                self.losses.append(loss.item())

                # log
                if step % self.n_intervals == self.n_intervals - 1:
                    self._log_loss(epoch, bch_idx)

            # test
            if epoch % self.n_validate_intervals == self.n_validate_intervals - 1:
                self.do_test(keyword=f"training :{epoch=}/{step=}")
        print("training done.")
        return self

    def _log_loss(self, epoch: int, bch_idx: int) -> Self:
        print(f"[{epoch+1}: {bch_idx+1}] / loss: {self.running_loss / self.n_intervals:.3f}")
        self.running_loss = 0.0
        return self

    def do_test(self, keyword="baseline") -> Self:
        n_corrects = 0
        n_totals = 0
        class_corrects = numpy.zeros(len(self.model.class_names), dtype=int)
        class_totals = numpy.zeros(len(self.model.class_names), dtype=int)
        total_losses = []

        print("=" * 120)
        for z in tqdm(self.testloader, desc="testloader"):
            X, t = self._t(z)

            with torch.no_grad():
                y = self.model(X)
                loss = self.model.loss(y, t)
            total_losses.append(loss.item())

            p = y.argmax(dim=-1)
            n_totals += len(t)  # add batch size, last is less batch size
            n_corrects += (p == t).sum().item()

            # stores by each label
            for n, ldx in enumerate(t.tolist()):
                assert ldx == t[n]
                class_totals[ldx] += 1
                class_corrects[ldx] += (p[n] == t[n]).sum().item()  # 一致してたら、+1

        print(f"test loss: {numpy.array(total_losses).mean():.3f}")
        print(f"accuracy[test][{keyword}]: {100 * n_corrects / n_totals: .3f} %")

        # NOTE: refactored
        for ldx, lbl in enumerate(self.model.class_names):
            print(f"{lbl} : {100 * class_corrects[ldx] / class_totals[ldx]: .3f} %")
        print("=" * 120)
        return self

    def get_latest_loss(self) -> float:
        return numpy.array(self.losses).mean()

    def save_model(self, model_file="data/model.pth") -> Self:
        joblib.dump(self.model, model_file)
        return self

    def load_model(self, model_file="data/model.pth") -> Self:
        self.model = joblib.load(model_file)
        return self
