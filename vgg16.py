from collections import namedtuple

import torch
from torchvision import models


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()

        # Get the pre-trained VGG16
        vgg_pretrained_features = models.vgg16(pretrained=True).features

        # The Sections of the VGG Network.
        self.section1 = torch.nn.Sequential()
        self.section2 = torch.nn.Sequential()
        self.section3 = torch.nn.Sequential()
        self.section4 = torch.nn.Sequential()

        # Add the layers.
        for current_layer in range(4):
            self.section1.add_module(str(current_layer), vgg_pretrained_features[current_layer])

        for current_layer in range(4, 9):
            self.section2.add_module(str(current_layer), vgg_pretrained_features[current_layer])

        for current_layer in range(9, 16):
            self.section3.add_module(str(current_layer), vgg_pretrained_features[current_layer])

        for current_layer in range(16, 23):
            self.section4.add_module(str(current_layer), vgg_pretrained_features[current_layer])

        # Set whether the parameters require the gradient.
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False


    def forward(self, X):
        current_activation = self.section1(X)
        section_relu1_2 = current_activation

        current_activation = self.section2(current_activation)
        section_relu2_2 = current_activation

        current_activation = self.section3(current_activation)
        section_relu3_3 = current_activation

        current_activation = self.section4(current_activation)
        section_relu4_3 = current_activation

        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        ret_val = vgg_outputs(section_relu1_2, section_relu2_2, section_relu3_3, section_relu4_3)

        return ret_val
