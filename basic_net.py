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

import utilities
from basic_net_model import BasicNet


def train_basic_net(args):
    # Train the Basic Net.
    
    # Set the Random Seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set the Random Seed in CUDA.
    if args.cuda:
        torch.cuda.manual_seed(args.seed)


    
    # Transform for loading the Target Image.
    load_target_image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    # Load the Target Image to a Batch.
    target_image = utilities.load_image(args.target_texture)
    target_image_tensor = load_target_image_transform(target_image)
    target_tensor_batch = target_image_tensor.repeat(args.batch_size, 1, 1, 1)

    # Move the Image over to CUDA.
    if args.cuda:
        target_tensor_batch.cuda()
    
    # Create the Variable and Normalize it.
    target_tensor_batch_v = Variable(target_tensor_batch)
    target_tensor_batch_v = utilities.normalize_image_batch(target_tensor_batch_v)

    # Create the BasicNet.
    basic_net = BasicNet()
    
    # Move the BasicNet over to CUDA.
    if args.cuda:
        basic_net.cuda()

    # Optimizer.
    optimizer = Adam(basic_net.parameters(), args.lr)

    # Loss Function.
    loss_function = torch.nn.MSELoss()

    # Go through the Iterations.
    for current_iteration in range(args.iterations):

        optimizer.zero_grad()
        current_input = Variable(torch.zeros(args.batch_size, 1, args.noise_size, args.noise_size))

        if args.cuda:
            current_input.cuda()
        

        

        



    

    return




def evaluate_basic_net(args):
    # Evaluate the Basic Net.


    return