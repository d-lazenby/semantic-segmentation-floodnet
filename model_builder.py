"""
Contains code to instantiate SegmentationModel class

Example:
  model = SegmentationModel(
    encoder_name='resnet101', 
    encoder_weights='imagenet,
    classes=10,
    activation=None)
"""

from torch import nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import JaccardLoss

class SegmentationModel(nn.Module):

    def __init__(self, encoder_name, encoder_weights, classes, activation):
        super(SegmentationModel, self).__init__()
        self.arc = smp.PSPNet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=classes,
        activation=activation
        )

    def forward(self, images, masks=None):

        logits = self.arc(images)

        if masks != None:
            loss = JaccardLoss(mode='multiclass')(logits, masks)
            return logits, loss

        return logits



