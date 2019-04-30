# import linear algebra and data manipulation libraries
import numpy as np
import pandas as pd

# import matplotlib for plotting
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

# import helper libraries
import requests
from io import BytesIO # Use When expecting bytes-like objects
import pickle
from collections import OrderedDict
import os
from os import path
import time
import argparse

# import PIL for image manipulation
from PIL import Image

# import machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# import pytorch
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms

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

def save_data(X_train, y_train, X_test, y_test):
    """
    The function saves datasets to disk as pickle files.

    INPUT:
        X_train - train dataset
        y_train - train dataset labels
        X_test - test dataset
        y_test - test dataset labels

    OUTPUT: None
    """
    print("Saving data \n")

    # Check for already saved files
    if not(path.exists('xtrain_doodle.pickle')):
        # Save X_train dataset as a pickle file
        with open('xtrain_doodle.pickle', 'wb') as f:
            pickle.dump(X_train, f)

        # Save X_test dataset as a pickle file
        with open('xtest_doodle.pickle', 'wb') as f:
            pickle.dump(X_test, f)

        # Save y_train dataset as a pickle file
        with open('ytrain_doodle.pickle', 'wb') as f:
            pickle.dump(y_train, f)

        # Save y_test dataset as a pickle file
        with open('ytest_doodle.pickle', 'wb') as f:
            pickle.dump(y_test, f)

def fit_model(model, X_train, y_train, epochs = 100, n_chunks = 1000, learning_rate = 0.003):
    """
    Function which fits the model.

    INPUT:
        model - pytorch model to fit
        X_train - (tensor) train dataset
        y_train - (tensor) train dataset labels
        epochs - number of epochs
        n_chunks - number of chunks to cplit the dataset
        learning_rate - learning rate value

    OUTPUT: None
    """

    print("Fitting model with epochs = {epochs}, learning rate = {lr}\n"\
    .format(epochs = epochs, lr = learning_rate))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    print_every = 100

    images = torch.chunk(X_train, n_chunks)
    labels = torch.chunk(y_train, n_chunks)

    steps = 0

    for e in range(epochs):
        running_loss = 0

        for i in range(n_chunks):
            steps += 1

            optimizer.zero_grad()

            # Forward and backward passes
            output = model.forward(images[i])
            loss = criterion(output, labels[i])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))

                running_loss = 0

def view_classify(img, ps):
    """
    Function for viewing an image and it's predicted classes
    with matplotlib.

    INPUT:
        img - (tensor) image file
        ps - (tensor) predicted probabilities for each class
    """
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(['cannon','eye', 'face', 'nail', 'pear','piano','radio','spider','star','sword'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()

    ts = time.time()
    plt.savefig('prediction' + str(ts) + '.png')

def save_model(model, input_size, output_size, hidden_sizes, filepath = 'checkpoint.pth'):
    """
    Functions saves model checkpoint.

    INPUT:
        model - pytorch model
        input_size - size of the input layer
        output_size - size of the output layer
        hidden_sizes - list of the hidden layer sizes
        filepath - path for the model to be saved to

    OUTPUT: None
    """

    print("Saving model to {}\n".format(filepath))
    checkpoint = {'input_size': input_size,
              'output_size': output_size,
              'hidden_layers': hidden_sizes,
              'state_dict': model.state_dict()}

    torch.save(checkpoint, filepath)

def load_model(filepath = 'checkpoint.pth'):
    """
    Function loads the model from checkpoint.

    INPUT:
        filepath - path for the saved model

    OUTPUT:
        model - loaded pytorch model
    """

    print("Loading model from {} \n".format(filepath))

    checkpoint = torch.load(filepath)
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    hidden_sizes = checkpoint['hidden_layers']
    model = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(hidden_sizes[1], hidden_sizes[2])),
                          ('relu3', nn.ReLU()),
                          ('logits', nn.Linear(hidden_sizes[2], output_size))]))
    model.load_state_dict(checkpoint['state_dict'])

    return model

def test_model(model, img):
    """
    Function creates test view of the model's prediction for image.

    INPUT:
        model - pytorch model
        img - (tensor) image from the dataset

    OUTPUT: None
    """

    # Convert 2D image to 1D vector
    img = img.resize_(1, 784)

    ps = get_preds(model, img)
    view_classify(img.resize_(1, 28, 28), ps)

def get_preds(model, input):
    """
    Function to get predicted probabilities from the model for each class.

    INPUT:
        model - pytorch model
        input - (tensor) input vector

    OUTPUT:
        ps - (tensor) vector of predictions
    """

    # Turn off gradients to speed up this part
    with torch.no_grad():
        logits = model.forward(input)
    ps = F.softmax(logits, dim=1)
    return ps

def get_labels(pred):
    """
        Function to get the vector of predicted labels for the images in
        the dataset.

        INPUT:
            pred - (tensor) vector of predictions (probabilities for each class)
        OUTPUT:
            pred_labels - (numpy) array of predicted classes for each vector
    """

    pred_np = pred.numpy()
    pred_values = np.amax(pred_np, axis=1, keepdims=True)
    pred_labels = np.array([np.where(pred_np[i, :] == pred_values[i, :])[0] for i in range(pred_np.shape[0])])
    pred_labels = pred_labels.reshape(len(pred_np), 1)

    return pred_labels

def evaluate_model(model, train, y_train, test, y_test):
    """
    Function to print out train and test accuracy of the model.

    INPUT:
        model - pytorch model
        train - (tensor) train dataset
        y_train - (numpy) labels for train dataset
        test - (tensor) test dataset
        y_test - (numpy) labels for test dataset

    OUTPUT:
        accuracy_train - accuracy on train dataset
        accuracy_test - accuracy on test dataset
    """
    train_pred = get_preds(model, train)
    train_pred_labels = get_labels(train_pred)

    test_pred = get_preds(model, test)
    test_pred_labels = get_labels(test_pred)

    accuracy_train = accuracy_score(y_train, train_pred_labels)
    accuracy_test = accuracy_score(y_test, test_pred_labels)

    print("Accuracy score for train set is {} \n".format(accuracy_train))
    print("Accuracy score for test set is {} \n".format(accuracy_test))

    return accuracy_train, accuracy_test

def plot_learning_curve(model, train, labels, y_train, test, y_test):
    """
    Function to plot learning curve depending on the number of epochs.

    INPUT:
        model - pytorch model
        train - (tensor) train dataset
        labels - (tensor) labels for train dataset
        y_train - (numpy) labels for train dataset
        test - (tensor) test dataset
        y_test - (numpy) labels for test dataset

    OUTPUT: None
    """
    train_acc = []
    test_acc = []

    for epochs in np.arange(10, 110, 10):
        # fit model
        fit_model(model, train, labels, epochs = epochs, n_chunks = 7000, learning_rate = 0.003)
        # get accuracy
        accuracy_train, accuracy_test = evaluate_model(model, train, y_train, test, y_test)

        train_acc.append(accuracy_train)
        test_acc.append(accuracy_test)

    # Plot curve
    x = np.arange(10, 110, 10)
    plt.plot(x, train_acc)
    plt.plot(x, test_acc)
    plt.legend(['train', 'test'], loc='upper left')
    plt.title('Accuracy depending on number of epochs', fontsize=20)
    plt.xlabel('Number of epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)

    ts = time.time()
    plt.savefig('learning_curve' + str(ts) + '.png')

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Argument parser')

    parser.add_argument('--save_dir', action='store', default = ' ',
                        help='Directory to save model checkpoint')

    parser.add_argument('--learning_rate', type = float, action='store', default = 0.003,
                        help='Model hyperparameters: learning rate')

    parser.add_argument('--epochs', type = int, action='store', default = 30,
                        help='Model hyperparameters: epochs')

    parser.add_argument('--gpu', action='store_true',
                        help='Run training on GPU')
    results = parser.parse_args()

    learning_rate = results.learning_rate
    epochs = results.epochs

    if (results.gpu == True):
        device = 'cuda'
    else:
        device = 'cpu'

    if (results.save_dir == ' '):
        save_path = 'checkpoint.pth'
    else:
        save_path = results.save_dir + '/' + 'checkpoint.pth'

    # Hyperparameters for our network
    input_size = 784
    hidden_sizes = [128, 100, 64]
    output_size = 10

    # Build a feed-forward network
    model = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(hidden_sizes[1], hidden_sizes[2])),
                          ('relu3', nn.ReLU()),
                          ('logits', nn.Linear(hidden_sizes[2], output_size))]))

    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # Save datasets to disk if required
    save_data(X_train, y_train, X_test, y_test)

    # Convert to tensors
    train = torch.from_numpy(X_train).float()
    labels = torch.from_numpy(y_train).long()
    test = torch.from_numpy(X_test).float()
    test_labels = torch.from_numpy(y_test).long()

    # Fit model
    fit_model(model, train, labels, epochs = epochs, n_chunks = 7000, learning_rate = learning_rate)
    plot_learning_curve(model, train, labels, y_train, test, y_test)

    # Evaluate model
    evaluate_model(model, train, y_train, test, y_test)

    # Save the model
    save_model(model, input_size, output_size, hidden_sizes, filepath = save_path)

if __name__ == '__main__':
    main()
