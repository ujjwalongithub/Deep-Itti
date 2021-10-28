import time

import pytorch_lightning as pl
import torch

from backbones import vgg


def get_backbone(backbone_name: str, num_classes: int) -> torch.nn.Module:
    """
    This function returns a backbone for classification.
    We currently use only the models defined in torchvision.models.
    :param backbone_name: Name of the backbone
    :param num_classes: Number of output classes
    :return: A torch.nn.Module object if backbone_name is reconized
    :raises: ValueError if backbone_name is not recognized
    """
    if backbone_name == 'vgg16':
        model = vgg.vgg16(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError('Unrecognized backbone_name')

    return model


class DeepIttiTrainer(pl.LightningModule):
    def __init__(self,
                 backbone_name: str,
                 num_classes: int
                 ):
        super(DeepIttiTrainer, self).__init__()
        self._backbone = get_backbone(backbone_name, num_classes)
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
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
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
