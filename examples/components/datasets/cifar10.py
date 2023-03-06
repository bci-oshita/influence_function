import torchvision
import torchvision.transforms as transforms

from .dataset import load_data_common

g_class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_data(
    target_train_indices: list = [],
    target_test_indices: list = [],
    bsz=64,
    do_shuffle=False,
):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    trainloader, testloader = load_data_common(
        dataclass=torchvision.datasets.CIFAR10,
        transform=transform,
        target_train_indices=target_train_indices,
        target_test_indices=target_test_indices,
        bsz=bsz,
        do_shuffle=do_shuffle,
    )

    return trainloader, testloader
