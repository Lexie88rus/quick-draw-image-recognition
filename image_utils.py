# import helpers
import numpy as np
import pandas as pd
import os
from os import path
import pickle
import random

# import matplotlib for plotting
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import imageio

# import PIL for image manipulation
from PIL import Image
from PIL import ImageOps

def load_data():
    """
    Function loads quick draw dataset. If no data is loaded yet, the datasets
    are loaded from the web. If there are already loaded datasets, then data
    is loaded from the disk (pickle files).

    INPUTS: None

    OUTPUT:
        X_train - train dataset
        y_train - train dataset labels
        X_test - test dataset
        y_test - test dataset labels
    """
    print("Loading data \n")

    # Check for already loaded datasets
    if not(path.exists('xtrain_doodle.pickle')):
        # Load from web
        print("Loading data from the web \n")

        # Classes we will load
        categories = ['cannon','eye', 'face', 'nail', 'pear','piano','radio','spider','star','sword']

        # Dictionary for URL and class labels
        URL_DATA = {}
        for category in categories:
            URL_DATA[category] = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/' + category +'.npy'

        # Load data for classes in dictionary
        classes_dict = {}
        for key, value in URL_DATA.items():
            response = requests.get(value)
            classes_dict[key] = np.load(BytesIO(response.content))

        # Generate labels and add labels to loaded data
        for i, (key, value) in enumerate(classes_dict.items()):
            value = value.astype('float32')/255.
            if i == 0:
                classes_dict[key] = np.c_[value, np.zeros(len(value))]
            else:
                classes_dict[key] = np.c_[value,i*np.ones(len(value))]

        # Create a dict with label codes
        label_dict = {0:'cannon',1:'eye', 2:'face', 3:'nail', 4:'pear',
                      5:'piana',6:'radio', 7:'spider', 8:'star', 9:'sword'}

        lst = []
        for key, value in classes_dict.items():
            lst.append(value[:3000])
        doodles = np.concatenate(lst)

        # Split the data into features and class labels (X & y respectively)
        y = doodles[:,-1].astype('float32')
        X = doodles[:,:784]

        # Split each dataset into train/test splits
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)
    else:
        # Load data from pickle files
        print("Loading data from pickle files \n")

        file = open("xtrain_doodle.pickle",'rb')
        X_train = pickle.load(file)
        file.close()

        file = open("xtest_doodle.pickle",'rb')
        X_test = pickle.load(file)
        file.close()

        file = open("ytrain_doodle.pickle",'rb')
        y_train = pickle.load(file)
        file.close()

        file = open("ytest_doodle.pickle",'rb')
        y_test = pickle.load(file)
        file.close()

    return X_train, y_train, X_test, y_test

def view_image(img, filename = 'image'):
    """
    Function to view numpy image with matplotlib.
    The function saves the image as png.
    INPUT:
        img - (numpy array) image from train dataset with size (1, 784)
        filename - name of a file where to save the image
    OUTPUT:
        None
    """
    fig, ax = plt.subplots(figsize=(6,9))
    ax.imshow(img.reshape(28, 28).squeeze())
    ax.axis('off')

    plt.savefig(filename + '.png')

def convert_to_PIL(img):
    """
    Function to convert numpy (1, 784) image to PIL image.
    INPUT:
        img - (numpy array) image from train dataset with size (1, 784)
    OUTPUT:
        pil_img - (PIL Image) 28x28 image
    """
    img_r = img.reshape(28,28)

    pil_img = Image.new('RGB', (28, 28), 'white')
    pixels = pil_img.load()

    for i in range(0, 28):
        for j in range(0, 28):
            if img_r[i, j] > 0:
                pixels[j, i] = (255 - int(img_r[i, j] * 255), 255 - int(img_r[i, j] * 255), 255 - int(img_r[i, j] * 255))

    return pil_img

def rotate_image(src_im, angle = 45, size = (28,28)):
    """
    Function to rotate PIL Image file
    INPUT:
        src_im - (PIL Image) 28x28 image to be rotated
        angle - angle to rotate the image
        size - (tuple) size of the output image
    OUTPUT:
        dst_im - (PIL Image) rotated image
    """
    dst_im = Image.new("RGBA", size, "white")
    src_im = src_im.convert('RGBA')

    rot = src_im.rotate(angle)
    dst_im.paste(rot, (0, 0), rot)

    return dst_im

def flip_image(src_im):
    """
    Function to flip a PIL Image file.
    INPUT:
        scr_im - (PIL Image) 28x28 image to be flipped
    OUTPUT:
        dst_im - (PIL Image) flipped image
    """
    dst_im = src_im.transpose(Image.FLIP_LEFT_RIGHT)
    return dst_im

def convert_to_np(pil_img):
    """
    Function to convert PIL Image to numpy array.
    INPUT:
        pil_img - (PIL Image) 28x28 image to be converted
    OUTPUT:
        img - (numpy array) converted image with shape (28, 28)
    """
    pil_img = pil_img.convert('RGB')

    img = np.zeros((28, 28))
    pixels = pil_img.load()

    for i in range(0, 28):
        for j in range(0, 28):
            img[i, j] = 1 - pixels[j, i][0] / 255

    return img

def add_flipped_and_rotated_images(X_train, y_train):
    """
    Function which adds flipped and rotated images to the original dataset.
    INPUT:
        X_train - (numpy array) the original training set
        y_train - (numpy array) original labels dataset
    OUTPUT:
        X_Train_new - (numpy array) the dataset with added flipped and rotated
        images
        y_train_new - (numpy array) labels for the new training dataset
    """
    print("Adding flipped and rotated images to the training set. \n")

    X_train_new = X_train.copy()
    y_train_new = y_train.copy().reshape(y_train.shape[0], 1)

    for i in range(0, X_train.shape[0]):
        # get image to rotate and flip
        img = X_train[i]
        pil_img = convert_to_PIL(img)

        # get random angle
        angle = random.randint(5, 10)

        # rotate and flip
        rotated = convert_to_np(rotate_image(pil_img, angle))
        flipped = convert_to_np(flip_image(pil_img))

        # add to the original dataset
        X_train_new = np.append(X_train_new, rotated.reshape(1, 784), axis = 0)
        X_train_new = np.append(X_train_new, flipped.reshape(1, 784), axis = 0)
        y_train_new = np.append(y_train_new, y_train[i].reshape(1,1), axis = 0)
        y_train_new = np.append(y_train_new, y_train[i].reshape(1,1), axis = 0)

        # print out progress
        if i % 100 == 0:
            print("Processed {i} files out of {total}.".format(i= i, total = X_train.shape[0]))

    return X_train_new, y_train_new

def view_label_heatmap(X_train, y_train, label, label_name):
    """
    Function to plot the heatmap for images with same label.
    INPUT:
        X_train - (numpy array) training dataset
        y_train - (numpy array) labels for training dataset
        label - (int) label for images
        label_name - (str) name for images label

    OUTPUT: None
    """
    # filter X_train to remove all other images
    label_filter = y_train == label
    X = pd.DataFrame(X_train)
    X_train_labeled = X[label_filter]

    # find mean value for pixels
    X_mean = np.sum(X, axis = 0).values

    # plot image
    fig, ax = plt.subplots(figsize=(6,9))
    ax.set_title(label_name)
    ax.imshow(X_mean.reshape(28, 28).squeeze())
    ax.axis('off')

    plt.savefig(label_name + '.png')

def view_images_grid(X_train, y_train, label, label_name):
    """
    Function to plot grid with several examples of images with
    passed label.
    INPUT:
        X_train - (numpy array) training dataset
        y_train - (numpy array) labels for training dataset
        label - (int) label for images
        label_name - (str) name for images label

    OUTPUT: None
    """
    indices = np.where(y_train == label)
    X = pd.DataFrame(X_train)

    for label_num in range(0,50):
        plt.subplot(5,10, label_num+1) #create subplots
        image = X.iloc[indices[0][label_num]].as_matrix().reshape(28,28)  #reshape images
        plt.imshow(image) #plot the data
        plt.xticks([]) #removes numbered labels on x-axis
        plt.yticks([]) #removes numbered labels on y-axis
        plt.suptitle(label_name)

        plt.savefig(label_name + '_grid.png')

def plot_image(image, label_name):
    """
    Helper function to plot 1 part of animated image.
    """
    fig, ax = plt.subplots(figsize=(5,5))

    plt.imshow(image) #plot the data
    plt.xticks([]) #removes numbered labels on x-axis
    plt.yticks([])

    ax.set_title(label_name)

    dims = (fig.canvas.get_width_height()[0] * 2, fig.canvas.get_width_height()[1] * 2)

    # Used to return the plot as an image array
    fig.canvas.draw() # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    #image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image  = image.reshape(dims[::-1] + (3,))

    return image

def create_animated_images(X_train, y_train, label, label_name):
    """
    Function creates animated gif with images of a certain label.
    INPUT:
        X_train - (numpy array) training dataset
        y_train - (numpy array) labels for training dataset
        label - (int) label for images
        label_name - (str) name for images label

    OUTPUT: None
    """
    # get images of a certain label
    indices = np.where(y_train == label)
    X = pd.DataFrame(X_train)

    images = []
    for label_num in range(0,50):
        image = X.iloc[indices[0][label_num]].as_matrix().reshape(28,28)  #reshape images
        images.append(image)

    # save plotted images into a gif
    kwargs_write = {'fps':1.0, 'quantizer':'nq'}
    imageio.mimsave('./'+ label_name + '.gif', [plot_image(i, label_name) for i in images], fps=1)

def crop_image(image):
    """
    Crops image (crops out white spaces).
    INPUT:
        image - PIL image of original size to be cropped
    OUTPUT:
        cropped_image - PIL image cropped to the center  and resized to (28, 28)
    """
    cropped_image = image

    # get image size
    width, height = cropped_image.size

    # get image pixels
    pixels = cropped_image.load()

    image_strokes_rows = []
    image_strokes_cols = []

    # run through the image
    for i in range(0, width):
        for j in range(0, height):
            # save coordinates of the image
            if (pixels[i,j][3] > 0):
                image_strokes_cols.append(i)
                image_strokes_rows.append(j)

    # if image is not empty then crop to contents of the image
    if (len(image_strokes_rows)) > 0:
        # find the box for image
        row_min = np.array(image_strokes_rows).min()
        row_max = np.array(image_strokes_rows).max()
        col_min = np.array(image_strokes_cols).min()
        col_max = np.array(image_strokes_cols).max()

        # find the box for cropping
        margin = min(row_min, height - row_max, col_min, width - col_max)

        # crop image
        border = (col_min, row_min, width - col_max, height - row_max)
        cropped_image = ImageOps.crop(cropped_image, border)

    # get cropped image size
    width_cropped, height_cropped = cropped_image.size

    # create square resulting image to paste cropped image into the center
    dst_im = Image.new("RGBA", (max(width_cropped, height_cropped), max(width_cropped, height_cropped)), "white")
    offset = ((max(width_cropped, height_cropped) - width_cropped) // 2, (max(width_cropped, height_cropped) - height_cropped) // 2)
    # paste to the center of a resulting image
    dst_im.paste(cropped_image, offset, cropped_image)

    #resize to 28,28
    dst_im.thumbnail((28,28), Image.ANTIALIAS)

    return dst_im

def normalize(arr):
    """
    Function performs the linear normalizarion of the array.
    https://stackoverflow.com/questions/7422204/intensity-normalization-of-image-using-pythonpil-speed-issues
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    INPUT:
        arr - orginal numpy array
    OUTPUT:
        arr - normalized numpy array
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr

def normalize_image(image):
    """
    Function performs the normalization of the image.
    https://stackoverflow.com/questions/7422204/intensity-normalization-of-image-using-pythonpil-speed-issues
    INPUT:
        image - PIL image to be normalized
    OUTPUT:
        new_img - PIL image normalized
    """
    arr = np.array(image)
    new_img = Image.fromarray(normalize(arr).astype('uint8'),'RGBA')
    return new_img

def alpha_composite(front, back):
    """Alpha composite two RGBA images.

    Source: http://stackoverflow.com/a/9166671/284318

    Keyword Arguments:
    front -- PIL RGBA Image object
    back -- PIL RGBA Image object

    """
    front = np.asarray(front)
    back = np.asarray(back)
    result = np.empty(front.shape, dtype='float')
    alpha = np.index_exp[:, :, 3:]
    rgb = np.index_exp[:, :, :3]
    falpha = front[alpha] / 255.0
    balpha = back[alpha] / 255.0
    result[alpha] = falpha + balpha * (1 - falpha)
    old_setting = np.seterr(invalid='ignore')
    result[rgb] = (front[rgb] * falpha + back[rgb] * balpha * (1 - falpha)) / result[alpha]
    np.seterr(**old_setting)
    result[alpha] *= 255
    np.clip(result, 0, 255)
    # astype('uint8') maps np.nan and np.inf to 0
    result = result.astype('uint8')
    result = Image.fromarray(result, 'RGBA')
    return result

def alpha_composite_with_color(image, color=(255, 255, 255)):
    """
    Helper function to convert RGBA to RGB.
    https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil

    Alpha composite an RGBA image with a single color image of the
    specified color and the same size as the original image.

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    """
    back = Image.new('RGBA', size=image.size, color=color + (255,))
    return alpha_composite(image, back)

def convert_to_rgb(image):
    """
    Converts RGBA PIL image into RGB image.
    INPUT:
        image - PIL RGBA image
    OUTPUT:
        image_rgb - PIL image converted to RGB
    """
    image_rgb = alpha_composite_with_color(image)
    image_rgb.convert('RGB')

    return image_rgb
