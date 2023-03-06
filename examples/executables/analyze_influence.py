import random

import joblib
import numpy
import torch
from tqdm import tqdm

from influence_function.influence import Influence, InfluenceCalculator, to_loader
from examples.components.models.model import Model
from examples.components.datasets import DatasetType

torch.manual_seed(12345)


def load_model(dataset_type: DatasetType):
    model: Model = joblib.load(f"data/model-{dataset_type.value}.pth")
    return model


def pickup_wrongs(model, testloader, device):
    # wrong_test_indices = {cnm: [] for cnm in g_class_names}
    wrong_test_indices = {}

    # ミニバッチ毎に処理
    for bch_idx, z_test_batch in enumerate(testloader):
        X_bch, t_bch = z_test_batch[0].to(device), z_test_batch[1].to(device)
        _y = model(X_bch)
        y = _y.argmax(dim=-1)

        assert len(y) == len(t_bch)

        n_totals = 0
        total_loss = 0
        # データ点毎に処理
        for n, ldx in enumerate(t_bch):
            assert ldx == t_bch[n]
            with torch.no_grad():
                loss = model.loss(_y[n], t_bch[n])

            total_loss += loss.item()
            n_totals += 1

            # 推定ミスのテストデータをlossとともに記録
            if y[n] != t_bch[n]:
                test_idx = bch_idx * testloader.batch_size + n
                wrong_test_indices[test_idx] = loss.item()

        loss_avg: float = total_loss / n_totals
    return wrong_test_indices, loss_avg


def analyze(
    dataset_type: DatasetType = DatasetType.cifar10,
    n_train_samples: int = -1,
    damping: float = 0.1,
    scaling: float = 25,
    r: int = 5000,
    n_influence_samples: int = -1,
    n_s_test_samples: int = 7,
    target_layers: list[str] = [],  # ["model.0"]
    top_n: int = 10,
) -> numpy.ndarray:
    if dataset_type == DatasetType.mnist:
        from ..components.datasets.mnist import g_class_names, load_data
        from ..components.models.model_mnist import SimpleModelMnist

        SimpleModel = SimpleModelMnist
    elif dataset_type == DatasetType.cifar10:
        from ..components.datasets.cifar10 import g_class_names, load_data
        from ..components.models.model_cifar10 import SimpleModelCifar10

        SimpleModel = SimpleModelCifar10
    else:
        raise NotImplementedError(f"{dataset_type.value=}")

    model: SimpleModel = load_model(dataset_type)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 影響関数値算出用にローダを取得
    trainloader, testloader = load_data(do_shuffle=False)

    # NOTE: loss が高い順にtop_n を抽出
    #       base_loss: テストデータセットすべての平均 loss
    #       base_loss は、使わない
    wrong_test_indices, __base_loss = pickup_wrongs(model, testloader, device)
    wrongs_top_n = sorted(wrong_test_indices.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # NOTE: 判定ミスした test_idx のみをロードするようにセットアップ
    sample_indices_test = [test_idx for test_idx, __loss in wrongs_top_n]
    sampleloader_test = to_loader(testloader.dataset, do_shuffle=False, indices=sample_indices_test)

    # # trainset のサンプリングは、高速化目的(ただし、精度は犠牲)
    if n_train_samples < 0:
        sample_indices_train = numpy.arange(len(trainloader.dataset))
        sample_trainset = trainloader.dataset
    else:
        sample_indices_train = random.sample(range(len(trainloader.dataset)), n_train_samples)
        sample_trainset = [z for idx, z in enumerate(trainloader.dataset) if idx in sample_indices_train]

    # # 影響関数値を算出
    ic = InfluenceCalculator(
        damping=damping,
        scaling=scaling,
        r=r,
        n_influence_samples=n_influence_samples,
        n_s_test_samples=n_s_test_samples,
    ).setup(model, sample_trainset, target_layers=target_layers, device=device)

    influence_results = {}
    for lbl in g_class_names:
        influence_results[lbl] = []  # initialize

    for test_idx, z_test in tqdm(zip(sample_indices_test, sampleloader_test), desc="sampleloader_test"):
        influ: Influence = ic.calculate_influences(test_idx, z_test)
        x_test, t_test = z_test
        lbl = g_class_names[int(t_test)]
        with torch.no_grad():
            p = model(x_test)
        test_estimation = p.cpu().numpy()
        influence_results[lbl].append(
            (
                influ.test_idx,
                influ.values,
                influ.helpful_train_indices,
                influ.harmful_train_indices,
                influ.values_index,
                test_estimation,
            )
        )

    return influence_results


def _main(
    dataset_type: DatasetType = DatasetType.cifar10,
    n_train_samples: int = -1,
    damping: float = 0.1,
    scaling: float = 25,
    r: int = 2,  # 5000
    n_influence_samples: int = 10,  # must be < n_train_samples     # -1
    n_s_test_samples: int = 16,  # must be < n_train_samples
    result_file: str = "result/analyzed_influence.gz",
):
    # 解析を実行
    results = analyze(
        dataset_type=dataset_type,
        n_train_samples=n_train_samples,
        damping=damping,
        scaling=scaling,
        r=r,
        n_influence_samples=n_influence_samples,
        n_s_test_samples=n_s_test_samples,
    )

    # 結果を出力
    joblib.dump(results, result_file, compress=("gzip", 3))

    return


if __name__ == "__main__":
    import typer

    typer.run(_main)
