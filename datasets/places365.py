import cv2
import torchvision.datasets as TDB

import datasets.augmentation as AUG


def places365_dataset_standard(
        root_dir: str,
        is_training: bool
):
    """
    Returns the Places-365 standard dataset from torchvision
    :param root_dir: Root directory of the Places365 dataset.
    :param is_training: True if in training mode else False.
    :return: A torchvision.Places365 object.
    """
    db = TDB.Places365(
        root=root_dir,
        split='train-standard' if is_training else 'val',
        transform=AUG.get_places365_transforms(is_training),
        loader=lambda x: cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB)
    )

    return db


def places365_dataset_challenge(
        root_dir: str,
        is_training: bool
):
    """
    Returns the Places-365 challenge dataset from torchvision
    :param root_dir: Root directory of the Places365 dataset.
    :param is_training: True if in training mode else False.
    :return: A torchvision.Places365 object.
    """
    db = TDB.Places365(
        root=root_dir,
        split='train-challenge' if is_training else 'val',
        transform=AUG.get_places365_transforms(is_training),
        loader=lambda x: cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB)
    )

    return db
