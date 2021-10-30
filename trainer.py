import multiprocessing as mp
import time

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.accelerators import GPUAccelerator
from pytorch_lightning.plugins import NativeMixedPrecisionPlugin, DDPPlugin
from pytorch_lightning.loggers import WandbLogger
import torchmetrics

from backbones import vgg
from datasets import imagenet


def get_backbone(backbone_name: str, num_classes: int, pretrained: bool =
False, replace_final_force: bool = False) \
        -> torch.nn.Module:
    """
    This function returns a backbone for classification.
    We currently use only the models defined in torchvision.models.
    :param backbone_name: Name of the backbone
    :param num_classes: Number of output classes
    :return: A torch.nn.Module object if backbone_name is reconized
    :raises: ValueError if backbone_name is not recognized
    """
    if backbone_name == 'vgg16':
        model = vgg.vgg16(pretrained=pretrained)
        if pretrained and replace_final_force:
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif backbone_name == 'vgg16_itti':
        model = vgg.vgg16(itti_koch=True, pretrained=pretrained)
        if pretrained and replace_final_force:
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError('Unrecognized backbone_name')

    return model


class DeepIttiTrainer(pl.LightningModule):
    def __init__(self,
                 backbone_name: str,
                 num_classes: int,
                 use_pretrained: bool = False,
                 replace_final_force: bool = False
                 ):
        super(DeepIttiTrainer, self).__init__()
        self._backbone = get_backbone(backbone_name, num_classes,
                                      pretrained=use_pretrained,
                                      replace_final_force=replace_final_force)
        self._softmax = torch.nn.Softmax()
        self._time_init = time.time()


    def forward(self, x):
        logits = self._backbone(x)
        probabilities = self._softmax(logits)
        return probabilities

    def training_step(self, batch, batch_idx, **kwargs):
        x, y = batch
        logits = self._backbone(x)
        loss = torch.nn.CrossEntropyLoss()(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        accuracy_top1 = torchmetrics.functional.accuracy(logits, y,
                                                      average='macro',
                                                    num_classes=1000,
                                                    top_k=1)
        self.log("train_accuracy_top1", accuracy_top1, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        accuracy_top5 = torchmetrics.functional.accuracy(logits, y,
                                                         average='macro',
                                                         num_classes=1000,
                                                         top_k=5)
        self.log("train_accuracy_top5", accuracy_top5, on_step=True,
                 on_epoch=True, prog_bar=True,
                 logger=True)
        accuracy_top10 = torchmetrics.functional.accuracy(logits, y,
                                                         average='macro',
                                                         num_classes=1000,
                                                         top_k=10)
        self.log("train_accuracy_top10", accuracy_top10, on_step=True,
                 on_epoch=True, prog_bar=True,
                 logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def on_train_epoch_start(self) -> None:
        self._time_init = time.time()
        return None

    def on_train_epoch_end(self, **kwargs) -> None:
        time_taken = time.time() - self._time_init
        self.log("Epoch time", time_taken, on_epoch=True, on_step=False)
        return None


if __name__ == "__main__":
    dataset = imagenet.imagenet_dataset(
        '/data/stars/share/STARSDATASETS/ILSVRC2012', is_training=True)
    train_loader = DataLoader(dataset=dataset, batch_size=32,
                              shuffle=True, num_workers=mp.cpu_count()-4,
                              pin_memory=True,
                              prefetch_factor=32)

    model = DeepIttiTrainer(backbone_name='vgg16_itti', num_classes=1000)

    accelerator = GPUAccelerator(
        precision_plugin=NativeMixedPrecisionPlugin(),
        training_type_plugin=DDPPlugin(find_unused_parameters=False),
    )


    logger = WandbLogger(name='vgg16_itti_imagenet',project='Deep-Itti',
                         entity='ujjwal', group="experiment_initial_itti_koch")
    trainer = pl.Trainer(gpus=3, strategy=DDPPlugin(
        find_unused_parameters=False),
                         gradient_clip_val=10.0,
                         logger=logger)

    trainer.fit(model, train_loader)
