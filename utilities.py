import torch
from torch.autograd import Variable
from PIL import Image

def load_image(filename):
    # Load and Return the Image.
    image = Image.open(filename)
    return image


def save_image(filename, data):
    # Save the Image.
    image = data
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

    input_batch = torch.div(input_batch / 255.0)
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


