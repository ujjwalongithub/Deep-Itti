import cv2
import torchvision.datasets as TDB

import datasets.augmentation as AUG


def imagenet_dataset(
        root_dir: str,
        is_training: bool
):
    """
      Returns the ILSVRC2012 dataset from torchvision
    :param root_dir: Root directory of the ImageNet Dataset.
    :param is_training:If True, creates dataset from training set, otherwise
    creates from validation set.
    :return: A torchvision.datasets.ImageNet object.
    """
    db = TDB.ImageNet(
        root=root_dir,
        split='train' if is_training else 'val',
        transform=AUG.get_imagenet_transforms(is_training),
        loader=lambda x: cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB)
    )

    return db
