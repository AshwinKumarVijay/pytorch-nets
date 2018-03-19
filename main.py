import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms



def main():
    
    main_argparser = argparse.ArgumentParser(description="parser for pytorch-basic-net")
    subparsers = main_argparser.add_subparsers(title="subcommands", dest="subcommand")

    # Add the Arguments for Training the Model.
    train_argparser = subparsers.add_subparsers("train",  help="parser for training arguments")

    # Add the Arguments for setup. CUDA and Seed.
    train_argparser.add_argument("--cuda", type=int, required=True, help="set it to 1 for running on GPU, 0 for CPU")    
    train_argparser.add_argument("--seed", type=int, default=1, help="random seed for training")


    # Add 
    





if __name__ == "__main__":
    main()