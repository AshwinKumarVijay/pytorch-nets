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

import diversified_net_script

def main():
    
    main_argparser = argparse.ArgumentParser(description="parser for pytorch-basic-net")
    subparsers = main_argparser.add_subparsers(title="subcommands", dest="subcommand")

    # ---- Arguments for Training the Model ---- #
    train_argparser = subparsers.add_parser("train",  help="Parser for Training Arguments")

    # Add the Arguments for setup. CUDA and Seed.
    train_argparser.add_argument("--cuda", type=int, required=True, help="Set it to 1 to use CUDA, 0 to use the CPU")    
    train_argparser.add_argument("--seed", type=int, default=1, help="Random seed for Training.")

    # Add the Arguments for the Training Target Texture.
    train_argparser.add_argument("--images-size", type=int, default=256, help="Resize the Target Texture.")
    train_argparser.add_argument("--target-texture", type=str, default="data/target_textures/brick.jpg", help="Path to Target Texture")

    # Add the Arguments for the Training Process.
    train_argparser.add_argument("--batch-size", type=int, default=20, help="Batch Size.")
    train_argparser.add_argument("--epochs", type=int, default=1, help="Number of Epochs")
    train_argparser.add_argument("--epoch-iterations", type=int, default=3000, help="Iterations per Epoch.")
    train_argparser.add_argument("--log-interval", type=int, default=25, help="Number of Images after which the Training Loss is logged.")

    # Add the Arguments for the Training Parameters.
    train_argparser.add_argument("--learning-rate", type=float, default=0.01, help="Learning Rate of the Model.")
    train_argparser.add_argument("--texture-loss-weight", type=float, default=1e10, help="Weight for the Diversity Loss.")
    train_argparser.add_argument("--diversity-loss-weight", type=float, default=-1e0, help="Weight for the Diversity Loss.")

    # Add the Arguments for the Training Output.
    train_argparser.add_argument("--save-model-path", type=str, default="data/trained_models/diversified_net/", help="Path to the folder in which to store the Trained Model.")
    train_argparser.add_argument("--save-model-filename", type=str, default="diversified_net.model", help="Name of the Trained Model.")


    # ---- Arguments for Evaluating the Model ---- #
    eval_argparser = subparsers.add_parser("eval", help="Parser for the Evaluation Arguments")

    # Add the Arguments for the Model to evaluate.
    eval_argparser.add_argument("--model-path", type=str, default="data/trained_models/diversified_net/diversified_net.model", help="Path to the Trained Model.")
    eval_argparser.add_argument("--batch-size", type=int, default=100)

    # Add the Arguments for setup. CUDA and Seed.
    eval_argparser.add_argument("--cuda", type=int, default=1, help="Set it to 1 to use CUDA, 0 to use the CPU")    
    eval_argparser.add_argument("--seed", type=int, default=1, help="Random seed for Training.")

    # Add the Arguments for the evaluation output.
    eval_argparser.add_argument("--output-path", type=str, default="data/texture_outputs/diversified_net/", help="Path to the Output.")
    eval_argparser.add_argument("--output-filename-prefix", type=str, default="texture", help="Name of the Output Texture.")

    
    # Parse the Arguments.
    args = main_argparser.parse_args()

    if args.subcommand is None:
        print("Error! Specify either Train or Evaluate.")
        sys.exit(1)

    if args.cuda and not torch.cuda.is_available():
        print("Error! CUDA is not available. Use the CPU instead.")
        sys.exit(1)


    if args.subcommand == "train":
        #  Train Basic Net.
        diversified_net_script.train_diversified_net(args)
    else:
        # Evaluate Basic Net.
        diversified_net_script.evaluate_diversified_net(args)




if __name__ == "__main__":
    main()