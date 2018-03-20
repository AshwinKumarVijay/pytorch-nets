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
    train_argparser = subparsers.add_subparsers("train",  help="Parser for Training Arguments")

    # Add the Arguments for setup. CUDA and Seed.
    train_argparser.add_argument("--cuda", type=int, required=True, help="Set it to 1 to use CUDA, 0 to use the CPU")    
    train_argparser.add_argument("--seed", type=int, default=1, help="Random seed for Training.")

    # Add the Arguments for the Training Target Texture.
    train_argparser.add_argument("--target-texture", type=str, default="data/target_textures/brick.jpg", help="Path to Target Texture")

    # Add the Arguments for the Training Data.
    train_argparser.add_argument("--batch-size", type=int, default=1, help="Batch Size.")
    train_argparser.add_argument("--input-size", type=int, default=1, help="Input Size.")

    # Add the Arguments for the Training Process.
    train_argparser.add_argument("--learning-rate", type=float, default=0.01, help="Learning Rate of the Model.")
    train_argparser.add_argument("--iterations", type=int, default=3000, help="Number of Iterations")
    train_argparser.add_argument("--diversity-weight", type=float, default=-1.0, help="Weight for the Diversity Loss.")
    train_argparser.add_argument("--texture-weight", type=float, default=-1.0, help="Weight for the Texture Loss.")

    

    
    # Parse the Arguments.
    args = main_argparser.parse_args()

    if args.subcommand is None:
        print("Error! Specify either Train or Evaluate.")
        sys.exit(1)

    if args.cuda and not torch.cuda.is_available():
        print("Error! CUDA is not available. Use the CPU instead.")
        sys.exit(1)


    if args.subcommand == "train"
        #  Train Basic Net.
        train_basic_net(args)
    else:
        # Evaluate Basic Net.
        evaluate_basic_net(args)




if __name__ == "__main__":
    main()