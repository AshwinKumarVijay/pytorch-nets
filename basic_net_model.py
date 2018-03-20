import argparse
import os
import sys
import time
from collections import namedtuple


import numpy as np
import torch
from torchvision import models


class BasicNet(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(BasicNet, self).__init__()


    def forward(self, X):

        return X

