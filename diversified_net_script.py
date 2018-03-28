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
from diversifed_net_model import DiversifiedNet

from vgg16 import Vgg16
from vgg19 import Vgg19


def train_diversified_net(args):
    # ---- Train the Diversified Net. ---- #


    # ---- Baisc Torch/CUDA Setup. ---- #
    # Set the Random Seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup CUDA and Set the Random Seed in CUDA.
    if args.cuda:
        torch.cuda.init()

        # Note the Device used.
        device_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)
        print("Using Device", device_index, "->", device_name)

        # Set the CUDA Random Seed.
        torch.cuda.manual_seed(args.seed)
    

    # ---- Load the Target Texture. ---- #
    # Transform for loading the Target Texture.
    load_target_texture_transform = transforms.Compose([
        transforms.Resize(args.images_size),
        transforms.CenterCrop(args.images_size),

        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    # Load the Target Style Image to a Batch.
    target_texture = utilities.load_image(args.target_texture)
    target_texture_tensor = load_target_texture_transform(target_texture)
    target_texture_tensor_batch = target_texture_tensor.repeat(args.batch_size, 1, 1, 1)

    # Move the Image Batch over to CUDA.
    if args.cuda:
        print("Moving the Target Texture to CUDA!")
        target_texture_tensor_batch = target_texture_tensor_batch.cuda()


    # Create the Variable and Normalize it.
    target_texture_tensor_batch_v = Variable(target_texture_tensor_batch)
    target_texture_tensor_batch_v = utilities.normalize_image_batch(target_texture_tensor_batch_v)
    
    


    # ---- Load the VGG Network ---- #
    # Create the VGG Network.
    vgg_network = Vgg16(requires_grad=False)

    # Move the VGG Network over to CUDA.
    if args.cuda:
        print("Moving the VGG Network to CUDA!")
        vgg_network.cuda()

    # ---- Get the Target Image Features and the Target Gram Features. ---- #
    target_texture_features = vgg_network(target_texture_tensor_batch_v)
    target_texture_gram_features = []

    # Compute the Zero Mean Activations at each of the layers.    
    for current_target_texture_feature in target_texture_features:

        # Compute the Zero Mean Target Texture Feature.
        zero_mean_target_texture_feature = utilities.zero_pixel_mean_image_batch(current_target_texture_feature)

        # Compute the Zero Gram Mean Feature.
        zero_mean_gram_feature = utilities.compute_gram_matrix(zero_mean_target_texture_feature)

        # Add the Target Texture.
        target_texture_gram_features.append(zero_mean_gram_feature)


    # ---- Texture Net to Train. ---- #
    # Create the DiversifiedNet.
    diversified_net = DiversifiedNet(2, 256, 3)

    if args.cuda:
        print("Moving the DiversifiedNet to CUDA!")
        diversified_net.cuda()

    # Optimizer and Loss Function.
    # Optimizer.
    optimizer = Adam(diversified_net.parameters(), args.learning_rate)

    # Loss Function.
    texture_loss_function = torch.nn.MSELoss()
    diversity_loss_function = torch.nn.MSELoss()

    # Set the Diversifed Net to be in Training Mode.
    diversified_net.train()

    # Go through the Epochs.
    for current_epoch in range(args.epochs):

        # Aggregate Texture and Diversity Loss.
        aggregate_texture_loss = 0
        aggregate_diversity_loss = 0
        current_epoch_inputs_count = 0


        for current_iteration_index in range(args.epoch_iterations):

            # Batch of Input Noise.
            inputs_batch = torch.rand(args.batch_size, 2, 1, 1)

            if args.cuda:
                inputs_batch = inputs_batch.cuda()

            # Add up the images that have been counted.
            current_epoch_inputs_count = current_epoch_inputs_count + len(inputs_batch)
            current_inputs_count = len(inputs_batch)

            # ---- Batch Setup ---- #
            # Zero out the Gradients.
            optimizer.zero_grad()

            # Get the Current Image Batch Variable
            current_image_batch_v = Variable(inputs_batch)

            # Move the Image Batch over to CUDA.
            if args.cuda:
                current_image_batch_v = current_image_batch_v.cuda()


            # ---- Forward Pass ---- #
            # Setup the Input.
            current_input_batch = current_image_batch_v

            # Get the Output.
            current_output_batch = diversified_net(current_input_batch)


            # ---- Compute Features ---- #
            # Compute the Normalized Current Output.
            normalized_current_output = utilities.normalize_image_batch(current_output_batch)
            
            # Compute the features of the Current Output.
            features_current_output = vgg_network(normalized_current_output)
            

            # ---- Texture Loss ---- #
            texture_loss = 0

            # Get the zero mean pixel value of the current_output.
            for current_output_feature, current_target_texture_gram_feature in zip(features_current_output, target_texture_gram_features):        
                
                # Compute the Zero Mean Features.
                zero_pixel_mean_current_feature = utilities.zero_pixel_mean_image_batch(current_output_feature)

                # Zero Mean Output Gram Features.
                zero_mean_output_gram_feature = utilities.compute_gram_matrix(zero_pixel_mean_current_feature)

                # Add to the texture loss.
                texture_loss = texture_loss + texture_loss_function(zero_mean_output_gram_feature, current_target_texture_gram_feature[:current_inputs_count, :, :])


            


            # ---- Diversity Loss ---- #
            diversity_loss = 0

            # Get the Current Dimensions.
            current_dimensions = features_current_output.relu4_3.size()

            # Set up the default order.
            current_order = list(range(current_dimensions[0]))

            # Shuffle the order.
            shuffled_order = utilities.fully_shuffled_order(current_order)

            # Compute the diversity loss.
            for current_order_index in range(len(current_order)):
                
                # Get the current output index.
                current_output_index = current_order[current_order_index]

                # Get the shuffled output index.
                shuffled_output_index = shuffled_order[current_order_index]

                # Compute the Current Diversity Loss.                
                current_diversity_loss = torch.sum(torch.abs(features_current_output.relu4_3[current_output_index] - features_current_output.relu4_3[shuffled_output_index])) / features_current_output.relu4_3[current_output_index].data.nelement() 
                
                # Add up the Total Diversity Loss.
                diversity_loss = diversity_loss + current_diversity_loss

            #
            diversity_loss = diversity_loss / len(current_order)

            # ---- Total Loss ---- #
            # Compute the total loss.
            total_loss = args.texture_loss_weight * texture_loss + args.diversity_loss_weight * diversity_loss
            total_loss.backward()


            # ---- Logging ---- #
            # Add to the Aggregate Content and Style Losses.
            aggregate_texture_loss = aggregate_texture_loss + args.texture_loss_weight * texture_loss.data[0]
            aggregate_diversity_loss = aggregate_diversity_loss + args.diversity_loss_weight * diversity_loss.data[0]

            #            
            if (current_iteration_index + 1) % args.log_interval == 0:                
                # Current Epoch and Current Batch.
                print("Current Epoch : ", current_epoch)
                print("Current Epoch Iteration : ", current_iteration_index + 1)

                # Current Content Loss and Style Loss.  
                print("Current Weighted Texture Loss", args.texture_loss_weight * texture_loss.data[0])
                print("Current Weighted Diversity Loss", args.diversity_loss_weight * diversity_loss.data[0])

                print("Current Total Loss", (aggregate_texture_loss + aggregate_diversity_loss)/(current_iteration_index + 1))
                

            # Step the Optimizer.
            optimizer.step()
                
        # Note the Current Epoch and the Total Loss. 
        print("Current Epoch : ", current_epoch)
        print("Aggregate Texture Loss :", aggregate_texture_loss)
        print("Aggregate Diversity Loss :", aggregate_diversity_loss)

        

    # Move the Transformer Net back to Evaluation.
    diversified_net.eval()

    # Move the Transformer Net to the CPU. 
    if args.cuda:
        diversified_net.cpu()

    # Check if the path to save it exists.
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    
    # Construct the save path and filename.
    save_model_path = os.path.join(args.save_model_path, args.save_model_filename)

    # Save the Model.
    torch.save(diversified_net.state_dict(), save_model_path)

    # Saved!
    print("Diversified Net Trained! Model Saved At :", save_model_path)

    return


def evaluate_diversified_net(args):
    # Evaluate the Diversified Net.


    # ---- Setup the Diversified Net ---- #
    # Create the Diversified Net.return
    diversified_net = DiversifiedNet(2, 256,3)
    diversified_net.load_state_dict(torch.load(args.model_path))

    # Move the DiversifiedNet over to CUDA.
    if args.cuda:
        diversified_net.cuda()

    
    # Batch of Input Noise.
    inputs_batch = torch.zeros(args.batch_size, 2, 1, 1)

    for current_input_index in range(args.batch_size):
        inputs_batch[current_input_index][0][0] =  (1.0 * current_input_index / (args.batch_size - 1))
        inputs_batch[current_input_index][1][0] =  1 - (1.0 * current_input_index / (args.batch_size - 1))

    if args.cuda:
        inputs_batch = inputs_batch.cuda()

    print(inputs_batch)

    inputs_batch = Variable(inputs_batch)

    # Get the Output.
    outputs_batch = diversified_net(inputs_batch)

    # Move the output back over the CPU.
    if args.cuda:
        outputs_batch = outputs_batch.cpu()

    #
    for current_element_index in range(args.batch_size):
        current_output_data = outputs_batch[current_element_index].data
                
        # Check if the path to save it exists.
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

        output_filename = args.output_filename_prefix + "_" + str(current_element_index) + ".jpg"

        output_path = os.path.join(args.output_path, output_filename)

        utilities.save_image(output_path, current_output_data)


    return 