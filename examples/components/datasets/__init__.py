from enum import Enum


class DatasetType(str, Enum):
    mnist = "mnist"
    cifar10 = "cifar10"
