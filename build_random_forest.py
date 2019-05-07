# import helpers
import numpy as np
import pandas as pd
import os
from os import path
import pickle

# import visualization
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# import machine learning
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

def fit_random_forest_classifier(X_train, y_train):
    '''
    INPUT:
        X_train - training set
        y_train - labels for training set
    OUTPUT:
        clf - trained Random Forest Classifier
    '''
    print("Training RF classifier.\n")
    # instantiate RF classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=None)

    # fit classifier
    clf.fit(X_train, y_train.ravel())

    return clf

def evaluate_random_forest_classifier(model, X_train, y_train, X_test, y_test):
    """
    Function returns test and train accyracy of the model.
    INPUT:
        model - fitted model
        X_train - training set
        y_train - labels for training set
        X_test - test set
        y_test - labels for test set

    OUTPUT:
        train_acc - training accuracy
        test_acc - test accuracy
    """
    #get predictions
    y_preds_train = model.predict(X_train)
    y_preds_test = model.predict(X_test)

    #find accuracy score
    train_acc = accuracy_score(y_train, y_preds_train)
    test_acc = accuracy_score(y_test, y_preds_test)

    return train_acc, test_acc

def scree_plot(pca):
    num_components=len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_

    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    for i in range(num_components):
        ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=12)

    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')
    plt.savefig('scree_plot.png')

def apply_PCA(n_components, X_train):
    '''
    Transforms data using PCA to create n_components, and provides back the results of the
    transformation.

    INPUT:
        n_components - int - the number of principal components to create
        data - dataset to transform

    OUTPUT:
        pca - the pca object created after fitting the data
        X_pca - the transformed X matrix with new number of components
    '''
    # scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(X_train)
    # apply PCA
    pca = PCA(n_components)
    X_pca = pca.fit_transform(X)
    return scaler, pca, X_pca

# load data
X_train, y_train, X_test, y_test = load_data()

# train RF classifier on original data and find out accuracy
print("RF classifier without PCA:\n")
model = fit_random_forest_classifier(X_train, y_train)
train_acc, test_acc = evaluate_random_forest_classifier(model, X_train, y_train, X_test, y_test)

print("Train acc: {train}, test acc: {test}.".format(train = train_acc, test = test_acc))

# train RF classifier on data after PCA
print("RF classifier with PCA:\n")
scaler, pca, X_train_pca = apply_PCA(700, X_train)
# plot the analysis of proncipal components
scree_plot(pca)
model = fit_random_forest_classifier(X_train_pca, y_train)
X_test_pca = scaler.fit_transform(X_test)
X_test_pca = pca.fit_transform(X_test_pca)
train_acc, test_acc = evaluate_random_forest_classifier(model, X_train_pca, y_train, X_test_pca, y_test)

print("Train acc: {train}, test acc: {test}.".format(train = train_acc, test = test_acc))
