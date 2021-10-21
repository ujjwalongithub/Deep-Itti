import torchvision.datasets as TDB

import datasets.augmentation as AUG


def cifar10(
        root_folder: str,
        is_training: bool = False,
        download: bool = True
) -> TDB.CIFAR10:
    """
    Returns the CIFAR10 dataset from torchvision
    :param root_folder: Root directory of dataset where directory
    cifar-10-batches-py exists or will be saved to if download is set to True.
    :param is_training:If True, creates dataset from training set, otherwise
    creates from test set.
    :param download:If true, downloads the dataset from the internet and puts
     it in root directory. If dataset is already downloaded, it is not
     downloaded again.
    :return: A torchvision.datasets.CIFAR10 object.
    """
    db = TDB.CIFAR10(
        root=root_folder,
        train=is_training,
        transform=AUG.get_cifar_transforms(is_training),
        download=download
    )
    return db


def cifar100(
        root_folder: str,
        is_training: bool = False,
        download: bool = True
) -> TDB.CIFAR100:
    """
        Returns the CIFAR100 dataset from torchvision
        :param root_folder: Root directory of dataset where directory
        cifar-10-batches-py exists or will be saved to if download is set to True.
        :param is_training:If True, creates dataset from training set, otherwise
        creates from test set.
        :param download:If true, downloads the dataset from the internet and puts
         it in root directory. If dataset is already downloaded, it is not
         downloaded again.
        :return: A torchvision.datasets.CIFAR100 object.
        """
    db = TDB.CIFAR100(
        root=root_folder,
        train=is_training,
        transform=AUG.get_cifar_transforms(is_training),
        download=download
    )
    return db
