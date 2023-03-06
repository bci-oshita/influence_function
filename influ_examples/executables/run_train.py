import torch
import torch.optim as optim

from influ_examples.components.datasets import DatasetType
from influ_examples.components.trainer import SimpleTrainer

torch.manual_seed(12345)


def _main(dataset_type: DatasetType = DatasetType.cifar10):
    if dataset_type == DatasetType.mnist:
        from influ_examples.components.datasets.mnist import g_class_names, load_data
        from influ_examples.components.models.model_mnist import SimpleModelMnist

        SimpleModel = SimpleModelMnist
    elif dataset_type == DatasetType.cifar10:
        from influ_examples.components.datasets.cifar10 import g_class_names, load_data
        from influ_examples.components.models.model_cifar10 import SimpleModelCifar10

        SimpleModel = SimpleModelCifar10
    else:
        raise NotImplementedError(f"{dataset_type.value=}")

    print(f"{g_class_names=}")

    # setup trainer
    model = SimpleModel(class_names=g_class_names)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.RAdam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    trainloader, testloader = load_data(do_shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainer = SimpleTrainer(model, optimizer, trainloader, testloader, device=device)

    trainer.do_train(max_epoch=10)
    trainer.do_test()
    trainer.save_model(model_file=f"data/model-{dataset_type.value}.pth")


if __name__ == "__main__":
    import typer

    typer.run(_main)
