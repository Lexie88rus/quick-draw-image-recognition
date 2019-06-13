# import linear algebra and data manipulation libraries
import numpy as np
import pandas as pd

# import helper libraries
import requests
from io import BytesIO # Use When expecting bytes-like objects
import pickle
from collections import OrderedDict
import os
from os import path
import time
import argparse
import wget
import ast

# import PIL for image manipulation
from PIL import Image, ImageDraw, ImageOps

# ndjson to open files
import ndjson

# import matplotlib for plotting
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

def load_categories(num_categories = 345, filepath = 'categories.txt'):
    '''
    Function loads the names of drawings categories from file.
    INPUT:
        1. num_categories - number of categories,
        2. filepath - fil, from which category names should be loaded.

    OUTPUT:
        1. categories - list of names of categories
        2. label_dict - dictionary with classes names and labels
    '''
    # read categories.txt and create a list of categories
    categories = []
    with open(filepath, 'r') as f:
        for i in range(0, num_categories):
            categories.append(f.readline()[:-1])

    # create a dictionary with labels for each category
    label_dict = {key:categories[key] for key in range(0,num_categories)}

    return categories, label_dict

def load_simplified_data(categories, data_filepath = './data'):
    '''
    Function loads simplified ndjson files for specified categories.
    INPUT:
        1. categories - list of names of categories to be downloaded
        2. data_filepath - folder name to store downloaded data
    '''

    # create a folder to store data if required
    if not os.path.exists(data_filepath):
        os.mkdir(data_filepath)

    # download files if required
    for category in categories: # check that file don't exist on the disk
        if not os.path.exists(data_filepath + '/' + str(category) + '.ndjson'):
            print("\nDownloading data for {}.".format(category))
            url = 'https://storage.googleapis.com/quickdraw_dataset/full/simplified/' + str(category) + '.ndjson'
            wget.download(
                        url=url,
                        out=data_filepath
                    )
        else:
            print("Data for {} is already downloaded.".format(category))

def convert_to_PIL(drawing, width = 256, height = 256):
    """
    Function to convert from drawing to PIL image.
    INPUT:
        drawing - drawing from 'drawing' column
        width - width of the initial image
        height - height of the initial image
    OUTPUT:
        pil_img - (PIL Image) image
    """

    # initialize empty (white) PIL image
    pil_img = Image.new('RGB', (width, height), 'white')
    pixels = pil_img.load()

    draw = ImageDraw.Draw(pil_img)

    # draw strokes as lines
    for x,y in drawing:
        for i in range(1, len(x)):
            draw.line((x[i-1], y[i-1], x[i], y[i]), fill=0)

    return pil_img

def convert_to_np_raw(drawing, width = 256, height = 256):
    """
    INPUT:
        drawing - drawing in initial format
        width - width of the initial image
        height - height of the initial image
    OUTPUT:
        img - drawing converted to the numpy array (width X height)
    """
    # initialize empty numpy array
    img = np.zeros((width, height))

    # create a PIL image out of drawing
    pil_img = convert_to_PIL(drawing)

    #resize to 28,28
    pil_img.thumbnail((width, height), Image.ANTIALIAS)

    pil_img = pil_img.convert('RGB')
    pixels = pil_img.load()

    # fill in numpy array with pixel values
    for i in range(0, width):
        for j in range(0, height):
            img[i, j] = 1 - pixels[j, i][0] / 255

    return img


def convert_to_np(pil_img, width = 256, height = 256):
    """
    Function to convert PIL Image to numpy array.
    INPUT:
        pil_img - (PIL Image) image to be converted
    OUTPUT:
        img - (numpy array) converted image with shape (width, height)
    """
    pil_img = pil_img.convert('RGB')

    img = np.zeros((width, height))
    pixels = pil_img.load()

    for i in range(0, width):
        for j in range(0, height):
            img[i, j] = 1 - pixels[j, i][0] / 255

    return img

def view_image(img, width = 256, height = 256, category = ''):
    """
    Function to view numpy image with matplotlib.
    The function saves the image as png.
    INPUT:
        img - (numpy array) image from train dataset with size (1, 784)
    OUTPUT:
        None
    """
    fig, ax = plt.subplots(figsize=(6,9))
    ax.imshow(img.reshape(width, height).squeeze())
    ax.axis('off')

    ts = time.time()
    plt.savefig('image_' +str(category) + '_' + str(ts) + '.png')

def get_drawings(category, data_filepath = './data', num_drawings = 0, width = 256, height = 256):
    '''
    Function to get drawing data out of ndjson file.
    INPUT:
        1. category - name of category to get data for
        2. data_filepath - path to folder containing ndjson files
        3. num_drawings - maximum number of drawings to get data for, all
        drawings are processed by default (num_drawings = 0)
        3. width, height - width and height of each image

    OUTPUT:
        1. np_images  - numpy images of specified category
    '''
    print("Preprocessing data for {}.\n".format(category))

    # open ndjson file containing image data
    filename = data_filepath + '/' + category + '.ndjson'
    with open(filename) as f:
        data = ndjson.load(f)

    # setup the number of images to fetch from file
    if num_drawings == 0:
        num_drawings = len(data)

    # convert images from raw to numpy
    np_images = []
    for i in range(min(len(data), num_drawings)):
        img = convert_to_PIL([ast.literal_eval(str(img)) for img in data[i]['drawing']], width = width, height = height)
        np_images.append(img)

    # return images array
    return np_images

def prepare_data(label_dict, data_filepath = './data', num_drawings = 0, width = 256, height = 256):
    """
    Function to get data for specified categories from the ndjson files
    and save it as preprocessed pickle file.
    INPUT:
        1. categories - name of categories to get data for
        2. data_filepath - path to folder containing ndjson files
        3. num_drawings - maximum number of drawings to get data for each
        category, all drawings are processed by default (num_drawings = 0)
        3. width, height - width and height of each image

    OUTPUT:
        None, the data is saved as npz
    """
    data = []
    labels = []
    # enumerate categories
    for key, value in label_dict.items():
        # get data for each category
        cat_data = get_drawings(value, data_filepath = data_filepath, num_drawings = num_drawings, width = width, height=height)
        [data.append(img) for img in cat_data]
        labels.append([key for i in range(num_drawings)])

    # convert to numpy array
    labels = np.array(labels)
    labels = labels.reshape(labels.shape[0] * labels.shape[1])

    # save data
    with open(data_filepath + '/' + 'data.pickle', 'wb') as f:
        pickle.dump(data, f)

    with open(data_filepath + '/' + 'labels.pickle', 'wb') as f:
        pickle.dump(labels, f)

def load_data(data_filepath = './data'):
    """
    Function loads data from pickle files.
    INPUT:
        1. data_filepath - path to stored preprocessed data
    OUTPUT:
        2. data, labels - loaded image data and labels
    """
    file = open(data_filepath + '/' + 'data.pickle','rb')
    data = pickle.load(file)
    file.close()

    file = open(data_filepath + '/' + 'labels.pickle','rb')
    labels = pickle.load(file)
    file.close()

    return data, labels

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Argument parser')

    parser.add_argument('--save_dir', action='store', default = './data',
                        help='Directory to save preprocessed data.')

    parser.add_argument('--num_classes', type = int, action='store', default = 10,
                        help='Number of classes to retreive.')

    parser.add_argument('--num_examples', type = int, action='store', default = 1000,
                        help='Number of training examples per class.')

    results = parser.parse_args()

    num_categories = results.num_classes
    num_examples = results.num_examples
    data_filepath = results.save_dir

    # load categories and create label dictionary
    categories, label_dict = load_categories(num_categories = num_categories)

    # load ndjson simplified files
    load_simplified_data(categories, data_filepath = data_filepath)

    # get images for apples
    #apples = get_drawings('apple', num_drawings = 5)

    # view image example
    #view_image(apples[3], width = 256, height = 256, category = 'apple')

    # prepare data and save to pickle files
    prepare_data(label_dict, data_filepath = data_filepath, num_drawings = num_examples, width = 256, height = 256)

    # load from pickle
    data, labels = load_data(data_filepath = data_filepath)

if __name__ == '__main__':
    main()
