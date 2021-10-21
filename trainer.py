import pytorch_lightning as pl
import torch
import torchvision.models as TM


def get_backbone(backbone_name: str, num_classes: int) -> torch.nn.Module:
    """
    This function returns an Itti-Koch backbone for classification
    :param backbone_name: Name of the backbone
    :param num_classes: Number of output classes
    :return: A torch.nn.Module object
    """
    if backbone_name == 'vgg16':
        model = TM.vgg16(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    elif backbone_name == 'alexnet':
        model = TM.alexnet(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif backbone_name == 'densenet169':
        model = TM.densenet169(pretrained=False)
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    elif backbone_name == 'resnext101_32x8d':
        model = TM.resnext101_32x8d(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif backbone_name == 'resnext':
        model = TM.resnext50_32x4d(pretrained=False)
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

    def forward(self, x):
        logits = self._backbone(x)
        probabilities = self._softmax(logits)
        return probabilities

    def training_step(self, batch, batch_idx, **kwargs):
        x, y = batch
        logits = self._backbone(x)
        loss = torch.nn.CrossEntropyLoss()(logits, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
