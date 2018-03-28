from collections import namedtuple

import torch
from torchvision import models


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()

        # Get the pre-trained VGG19
        vgg_pretrained_features = models.vgg19(pretrained=True).features

    def forward(self, X):

        return X
