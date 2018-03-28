import random
import torch
from torch.autograd import Variable
from PIL import Image

def load_image(filename):
    # Load and Return the Image.
    image = Image.open(filename)
    return image


def save_image(filename, data):
    # Save the Image.
    image_data = data.clone().clamp(0, 255).numpy()
    image_data = image_data.transpose(1, 2, 0).astype("uint8")
    image = Image.fromarray(image_data)
    image.save(filename)



def compute_gram_matrix(input_batch):
    # Compute the gram matrix of the input batch.
    (b, ch, h, w) = input_batch.size()
    features = input_batch.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram_matrix = features.bmm(features_t) / (ch * h * w)
    return gram_matrix


def normalize_image_batch(input_batch):
    # Normalize the Batch using the ImageNet Mean and STD.
    mean = input_batch.data.new(input_batch.data.size())
    std = input_batch.data.new(input_batch.data.size())

    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406

    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225

    input_batch = torch.div(input_batch, 255.0)
    input_batch -= Variable(mean)
    input_batch = input_batch / Variable(std)

    return input_batch


def normalize_image(input_image):
    # Normalize the Image using the ImageNet Mean and STD.
    mean = input_image.data.new(input_image.size())
    std = input_image.data.new(input_image.size())

    mean[0, :, :] = 0.485
    mean[1, :, :] = 0.456
    mean[2, :, :] = 0.406

    std[0, :, :] = 0.229
    std[1, :, :] = 0.224
    std[2, :, :] = 0.225

    input_image = torch.div(input_image, 255.0)
    input_image -= Variable(mean)
    input_image = input_image / Variable(std)

    return input_image


def zero_pixel_mean_image_batch(input_batch):
    # Zero out the Pixel Mean of the Image Batch. (Batch * Channels * Width * Height)
    input_batch_size = input_batch.size()

    # Get a view of all the pixels in each of the channels.
    input_batch_view = input_batch.view(input_batch_size[0], input_batch_size[1], input_batch_size[2] * input_batch_size[3])

    # Compute the mean.
    input_batch_view_mean = input_batch_view.mean(2)

    # Reshape the view to a tensor.
    input_batch_view_mean = input_batch_view_mean.unsqueeze(2).unsqueeze(3)

    # Get the Input Batch Mean.
    input_batch_mean = input_batch_view_mean.repeat(1, 1, input_batch_size[2], input_batch_size[3])

    # Return the Zero Pixel Mean Image Batch.
    return torch.div(input_batch - input_batch_mean, input_batch_size[0])


  
    

def fully_shuffled_order(input_order):
    # Shuffle the order till all the indices are different.
    copy_order = input_order[:]

    # Keep going until all the indices are different.
    while True:

        # Shuffle.
        random.shuffle(copy_order)
        
        # Reset the condition.
        is_shuffled = True

        # Go through the order.
        for i in range(len(input_order)):

            # Check if indices match.
            if input_order[i] == copy_order[i]:
                is_shuffled = False
                break

        # Check if we are shuffled.
        if is_shuffled:
            break

    # Return the copy order.    
    return copy_order
            








