import albumentations as A
import albumentations.pytorch.transforms as AP
import cv2


def get_imagenet_transforms(is_training: bool = False) -> A.Compose:
    """
    Returns an image augmentation transform for ILSVRC-2012 dataset for training
    or validation/testing
    :param is_training: A bool which is true for training and false otherwise.
    :return: An albumentations.Compose object.
    """
    if is_training:
        return _imagenet_train()
    else:
        return _imagenet_val()


def get_cifar_transforms(is_training: bool = False) -> A.Compose:
    """
    Returns an image augmentation transform for CIFAR10/100 dataset for training
    or validation/testing. Use this function for both CIFAR 10 and CIFAR 100
    datasets.
    :param is_training: A bool which is true for training and false otherwise.
    :return: An albumentations.Compose object.
    """
    if is_training:
        return _cifar_train()
    else:
        return _cifar_val()


def _imagenet_train() -> A.Compose:
    """
    Image augmentation transform for ILSVRC-2012 dataset during training.
    :return: An albumentations.Compose object
    """
    transform = A.Compose(
        [
            A.SmallestMaxSize(max_size=256,
                              interpolation=cv2.INTER_CUBIC,
                              p=1.0),
            A.RandomCrop(height=224, width=224, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.ToGray(p=0.2),
            A.RandomBrightnessContrast(p=0.5),
            A.CoarseDropout(p=0.3),
            AP.ToTensorV2(p=1.0)
        ]
    )

    return transform


def _imagenet_val() -> A.Compose:
    """
    Image augmentation transforms for ILSVRC-2012 dataset during
    validation/testing.
    :return: An albumentations.Compose object
    """

    transform = A.Compose(
        [
            A.SmallestMaxSize(max_size=256, interpolation=cv2.INTER_CUBIC,
                              p=1.0),
            A.CenterCrop(height=225, width=224, p=1.0),
            AP.ToTensorV2(p=1.0)
        ]
    )
    return transform


def _cifar_train() -> A.Compose:
    """
    Image augmentation transform for CIFAR10/100 dataset during training.
    :return: An albumentations.Compose object
    """
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ToGray(p=0.2),
            A.RandomBrightnessContrast(p=0.5),
            A.CoarseDropout(max_holes=3, max_height=3, max_width=3, p=0.5),
            AP.ToTensorV2(p=1.0)
        ]
    )
    return transform


def _cifar_val() -> A.Compose:
    """
    Image augmentation transform for CIFAR10/100 dataset during
    validation/testing.
    :return: An albumentations.Compose object
    """
    transform = A.Compose(
        [
            AP.ToTensorV2(p=1.0)
        ]
    )
    return transform
