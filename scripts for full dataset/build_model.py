# import linear algebra and data manipulation libraries
import numpy as np
import pandas as pd

# import PIL for image manipulation
from PIL import Image, ImageDraw, ImageOps

# import matplotlib for plotting
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

# import helper libraries
from collections import OrderedDict
import os
from os import path
import time
import argparse

# import pytorch
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader

# import machine learning libraries
from sklearn.model_selection import train_test_split

# import utilities to load the dataset
from load_data import load_categories, load_data

def convert_to_PIL(img, width = 256, height = 256):
    """
    Function to convert numpy (width, height) image to PIL image.
    INPUT:
        img - (numpy array) image from train dataset with size (width, height)
    OUTPUT:
        pil_img - (PIL Image) (width x height) image
    """
    img_r = img.reshape(width, height)

    pil_img = Image.new('RGB', (width, height), 'white')
    pixels = pil_img.load()

    for i in range(0, width):
        for j in range(0, height):
            if img_r[i, j] > 0:
                pixels[j, i] = (255 - int(img_r[i, j] * 255), 255 - int(img_r[i, j] * 255), 255 - int(img_r[i, j] * 255))

    return pil_img

class DoodlesDataset(Dataset):
    """
    Dataset class for doodles.
    """

    def __init__(self, data, labels, width = 256, height = 256, transform=None):
        """
        The initialization method.
        INPUT:
            1. data - images data
            2. labels - labels data
            3. width - image width and height
            4. transform (callable, optional) - Optional transform to be applied
                on a sample
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return [image, label]

def initialize_model(num_categories):
    """
    Function returns machine learning model to be trained.
    INPUT:
        1. num_categories - number of classes
    OUTPUT:
        1. model - resulting model to be trained
    """
    model = models.densenet169(pretrained=False)
    input_size = 1664
    hidden_sizes = [512, 128, 64]
    model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                          ('bn2', nn.BatchNorm1d(num_features=hidden_sizes[1])),
                          ('relu2', nn.ReLU()),
                          ('dropout', nn.Dropout(0.3)),
                          ('fc3', nn.Linear(hidden_sizes[1], hidden_sizes[2])),
                          ('bn3', nn.BatchNorm1d(num_features=hidden_sizes[2])),
                          ('relu3', nn.ReLU()),
                          ('logits', nn.Linear(hidden_sizes[2], num_categories))]))

    return model

def train_model(model, trainloader, testloader, epochs, batch_size, learning_rate, device):
    '''
    Function to train the model.
    INPUT:
        1. model - model to train
        2. trainloader - data loader for training data
        3. testloader - data loader for test data
        4. epochs - number of epochs
        5. batch_size - size of the batch
        6. learning_rate - learning rate
        7. device - cuda or gpu
    '''
    # set loss function
    criterion = nn.NLLLoss()

    # set optimizer, only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train the model
    model.to(device)

    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                roc_auc = 0

                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy and roc_auc
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(testloader):.3f}.. ")
                running_loss = 0
                model.train()

def main():
    print('Load data: \n')

    # load categories and labels dictionary
    categories, labels_dict = load_categories(num_categories = 10)

    # load from pickle
    data, labels = load_data(data_filepath = './data')

    print('Setup the model: \n')
    # define train transformations
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5],
                                                                [0.5, 0.5, 0.5])])
    # define test transformations
    test_transforms = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # Split each dataset into train/test splits
    X_train, X_test, y_train, y_test = train_test_split(data,labels,test_size=0.1,random_state=42)

    # setup datasets and transformations
    train_dataset = DoodlesDataset(X_train, y_train, transform=train_transforms)
    test_dataset = DoodlesDataset(X_test, y_test, transform=test_transforms)

    # setup dataloaders
    trainloader = DataLoader(train_dataset, batch_size=128,
                        shuffle=True, num_workers=0)

    testloader = DataLoader(test_dataset, batch_size=32,
                            shuffle=True, num_workers=0)

    # setup Hyperparameters
    epochs = 1
    batch_size = 128
    learning_rate = 0.001
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # setup the model
    model = initialize_model(10)

    print('Train the model: \n')
    # train the model
    train_model(model, trainloader, testloader, epochs, batch_size, learning_rate, device)

if __name__ == '__main__':
    main()
