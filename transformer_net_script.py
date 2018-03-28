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
from transformer_net_model import TransformerNet

from vgg16 import Vgg16
from vgg19 import Vgg19


def train_transformer_net(args):
    # Train the Transformer Net.
    
    # Set the Random Seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set the Random Seed in CUDA.
    if args.cuda:
        torch.cuda.init()
        torch.cuda.manual_seed(args.seed)

        device_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)

        print("Using Device", device_index, "->", device_name)


    # ---- Load the Target Style Image. ---- #
    # Transform for loading the Target Style Image.
    load_target_style_image_transform = transforms.Compose([
        transforms.Resize(args.images_size),
        transforms.CenterCrop(args.images_size),

        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    # Load the Target Style Image to a Batch.
    target_style_image = utilities.load_image(args.target_style_image)
    target_style_image_tensor = load_target_style_image_transform(target_style_image)
    target_style_tensor_batch = target_style_image_tensor.repeat(args.batch_size, 1, 1, 1)
    
    # Move the Image over to CUDA.
    if args.cuda:
        print("Moving the Target Texture to CUDA!")
        target_style_tensor_batch = target_style_tensor_batch.cuda()
    
    # Create the Variable and Normalize it.
    target_style_tensor_batch_v = Variable(target_style_tensor_batch)
    target_style_tensor_batch_v = utilities.normalize_image_batch(target_style_tensor_batch_v)


    # ---- Load the Training Images. ---- #
    # Transform the Training Images to the Expected Size.
    load_training_image_transform = transforms.Compose([
        transforms.Resize(args.images_size),
        transforms.CenterCrop(args.images_size),

        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    # Get the Training Dataset.
    train_dataset = datasets.ImageFolder(args.training_dataset, load_training_image_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)


    # ---- Load the VGG Network. ---- # 
    # Create the VGG Network.
    vgg_network = Vgg16(requires_grad=False)

    # Move the VGG Network over to CUDA.
    if args.cuda:
        print("Moving the VGG Network to CUDA!")
        vgg_network.cuda()

    # ---- Get the Target Image Features and the Target Gram Features. ---- #
    target_style_features = vgg_network(target_style_tensor_batch_v)
    target_style_gram_features = [utilities.compute_gram_matrix(current_target_style_feature) for current_target_style_feature in target_style_features]    


    # ---- Texture Net to Train. ---- #
    # Create the TransformerNet.
    transformer_net = TransformerNet()
    
    # Move the TransformerNet over to CUDA.
    if args.cuda:
        print("Moving the TransformerNet to CUDA!")
        transformer_net.cuda()


    # Optimizer and Loss Function.
    # Optimizer.
    optimizer = Adam(transformer_net.parameters(), args.learning_rate)

    # Loss Function.
    loss_function = torch.nn.MSELoss()

    # Set the Transformer Net to be in Training Mode.
    transformer_net.train()

    # Go through the Epochs.
    for current_epoch in range(args.epochs):

        # Aggregate Content and Style Loss.
        aggregate_content_loss = 0
        aggregate_style_loss = 0
        current_epoch_images_count = 0

        # Go through each batch.
        for current_batch_index, (current_image_batch, _) in enumerate(train_loader):

            # Add up the images that have been counted.
            current_epoch_images_count = current_epoch_images_count + len(current_image_batch)
            current_images_count = len(current_image_batch)

            # ---- Batch Setup ---- #
            # Zero out the Gradients.
            optimizer.zero_grad()

            # Get the Current Image Batch Variable
            current_image_batch_v = Variable(current_image_batch)

            # Move the Image Batch over to CUDA.
            if args.cuda:
                current_image_batch_v = current_image_batch_v.cuda()


            # ---- Forward Pass ---- #
            # Setup the Input.
            current_input = current_image_batch_v

            # Get the Output.
            current_output = transformer_net(current_input)

            
            # ---- Compute Features ---- #
            # Compute the Normalized Current Output.
            normalized_current_output = utilities.normalize_image_batch(current_output)
            
            # Compute the features of the Current Output.
            features_current_output = vgg_network(normalized_current_output)

            # Normalize the Current Image Batch.
            current_image_batch_v = utilities.normalize_image_batch(current_image_batch_v)

            # Compute the features of the Current Image Batch
            features_current_image_batch = vgg_network(current_image_batch_v)



            # ---- Content Loss ---- #
            # Compute the Content Loss.
            content_loss = loss_function(features_current_output.relu2_2, features_current_image_batch.relu2_2)


            # ---- Style Loss ---- #
            # Compute the Style Loss.
            style_loss = 0

            # Compute the loss between the gram features of each of the style loss layers.
            for current_output_feature, current_target_style_gram_feature in zip(features_current_output, target_style_gram_features):
                
                # Compute the Output Gram Feature.
                current_output_gram_feature = utilities.compute_gram_matrix(current_output_feature)

                # Compute the Style Loss between the Gram Features.
                style_loss += loss_function(current_output_gram_feature, current_target_style_gram_feature[:current_images_count, :, :])


            # ---- Total Loss ---- #
            # Compute the total loss.
            total_loss = args.content_loss_weight * content_loss + args.style_loss_weight * style_loss
            total_loss.backward()


            # ---- Logging ---- #
            # Add to the Aggregate Content and Style Losses.
            aggregate_content_loss = aggregate_content_loss + args.content_loss_weight * content_loss.data[0]
            aggregate_style_loss = aggregate_style_loss + args.style_loss_weight * style_loss.data[0]

            if (current_batch_index + 1) % args.log_interval == 0:
                # Current Epoch and Current Batch.
                print("Current Epoch : ", current_epoch)
                print("Current Batch : ", current_batch_index + 1)

                # Current Content Loss and Style Loss.
                print("Current Weighted Content Loss", aggregate_content_loss/(current_batch_index + 1))
                print("Current Weighted Style Loss", aggregate_style_loss/(current_batch_index + 1))
                print("Current Total Loss", (aggregate_content_loss + aggregate_style_loss)/(current_batch_index + 1))
                

            # Step the Optimizer.
            optimizer.step()
                
        # Note the Current Epoch and the Total Loss. 
        print("Current Epoch : ", current_epoch)
        print("Aggregate Content Loss :", aggregate_content_loss)
        print("Aggregate Style Loss :", aggregate_style_loss)



    # Move the Transformer Net back to Evaluation.
    transformer_net.eval()

    # Move the Transformer Net to the CPU. 
    if args.cuda:
        transformer_net.cpu()

    # Check if the path to save it exists.
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    
    # Construct the save path and filename.
    save_model_path = os.path.join(args.save_model_path, args.save_model_filename)

    # Save the Model.
    torch.save(transformer_net.state_dict(), save_model_path)

    # Saved!
    print("Transformer Net Trained! Model Saved At :", save_model_path)
    
    return



def evaluate_transformer_net(args):
    # Evaluate the Transformer Net.

    # ----  Setup the Transformer Net ---- #
    # Create the Transformer Net.
    transformer_net = TransformerNet()
    transformer_net.load_state_dict(torch.load(args.model_path))

    # Move the TransformerNet over to CUDA.
    if args.cuda:
        transformer_net.cuda()

    # ---- Load the Image ---- #
    load_image_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),

        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    # Setup the Input.
    transformed_image = load_image_transform(utilities.load_image(args.input_image_path))
    current_input = transformed_image.repeat(1, 1, 1, 1)

    # Move the input over to CUDA.
    if args.cuda:
        current_input = current_input.cuda()

    # Get the Output.
    current_output = transformer_net(current_input)

    # Move the output back over the CPU.
    if args.cuda:
        current_output = current_output.cpu()

    # Get the Output Data.
    output_data = current_output.data[0]

    # Check if the path to save it exists.
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Construct the Output Path.
    output_path = os.path.join(args.output_path, args.output_filename)

    # Save the Image.
    utilities.save_image(output_path, output_data)

    return