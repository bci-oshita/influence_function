import numpy
import torch
import torchvision.transforms as transforms
from torchvision.datasets.vision import VisionDataset


def _setup_dataset(dataset, indices):
    dataset.data = dataset.data[indices]
    dataset.targets = list(numpy.array(dataset.targets)[indices])
    return


def load_data_common(
    dataclass: type[VisionDataset],
    transform: transforms.Compose,
    target_train_indices: list = [],
    target_test_indices: list = [],
    bsz=64,
    do_shuffle=False,
):
    # load raw data
    # # trainset
    trainset = dataclass(root="./data", train=True, download=True, transform=transform)
    # # testset
    testset = dataclass(root="./data", train=False, download=True, transform=transform)
    # setup randomizer
    rs = numpy.random.RandomState(12345)
    train_indices = rs.permutation(range(len(trainset.data)))
    test_indices = rs.permutation(range(len(testset.data)))
    _setup_dataset(trainset, train_indices)
    _setup_dataset(testset, test_indices)

    # setup filter
    # NOTE: must be after randomized dataset
    if target_train_indices:
        _setup_dataset(trainset, target_train_indices)

    if target_test_indices:
        _setup_dataset(testset, target_test_indices)

    print(f"{trainset.targets[:20]=}")
    print(f"{testset.targets[:20]=}")

    # build loader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsz, shuffle=do_shuffle, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=bsz, shuffle=do_shuffle, num_workers=2)

    return trainloader, testloader
