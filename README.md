# Quick, Draw! Image Recognition
Recognition of Quick, Draw! game doodles.

## Table of Contents
* [Definition](#definition)
* [Input Data](#input-data)
* [Implementation](#implementation)
* [Repository Structure](#repository-structure)
* [Conclusions](#conclusions)

## DEFINITION
### Project Overview
[The Quick Draw Dataset](https://github.com/googlecreativelab/quickdraw-dataset) is a collection of 50 million drawings across 345 categories, contributed by players of the game Quick, Draw!. The player starts with an object to draw (for example it may say "Draw a chair in under 20 seconds"). Then the player has twenty seconds to draw that object. Based on what they draw, the AI guesses what they are drawing.
Research in recognition of images drawn by humans can improve pattern recognition solutions more broadly. Improving pattern recognition has an impact on handwriting recognition and its robust applications in areas including OCR (Optical Character Recognition), ASR (Automatic Speech Recognition) & NLP (Natural Language Processing).
In this project I analyzed the drawings and tried to build a deep learning application to classify those drawings.
### Problem Statement
Recognition of a drawing is a classification problem. I have to build a solution, which classifies input images. I split the whole problem of recognition of drawings into the following tasks:
* Input data analysis and preprocessing;
* Building a model to classify drawings;
* Evaluation of the model concerning chosen metrics;
* Building a web-application to demonstrate the results.
### Metrics
I chose accuracy as a metric to evaluate the results. Because of the rules of the game, we mostly care about how many times did the AI recognize the drawing correctly, and this is just the accuracy of the model.
## Input Data
I chose the simplified dataset with images in .npy format. This format is the easiest to use, preprocess, and to produce by a web application.
The examples of simplified images from the dataset:
![images examples](https://github.com/Lexie88rus/quick-draw-image-recognition/blob/master/assets/image%20grids/eye_grid.png)
## Implementation
Since there is a lot of data, and I can even generate additional data by flipping and rotating the images, I decided to use deep learning approaches to classify drawings.
The implementation consists of two parts:
1.	Building and refining the deep learning model to classify drawings;
2.	Building a web application to demonstrate the model.
### Data Preprocessing
I chose already preprocessed dataset with images which are already cropped and resized to 28 to 28 pixels. However, I decided also to generate a set of slightly rotated and flipped images. I added these images to the training dataset to reduce the variance of the resulting model.

<p>Image preprocessing example:

![preprocessing](https://github.com/Lexie88rus/quick-draw-image-recognition/blob/master/assets/image%20examples/preprocessing_example.png)
   
### Building the Model
To simplify the task a little I chose only ten image classes from the initial dataset.
My goal is to build a model, which takes 28 x 28 pixels image as an input and gives probabilities for each of the possible classes as an output. The figure below demonstrates the desired result:

![result example](https://github.com/Lexie88rus/quick-draw-image-recognition/blob/master/assets/prediction_eye.png)

I started with a simple neural network with two hidden layers built with the PyTorch library.
The sizes of the layers are as follows:
* Input layer: 784 (for 28 x 28 images),
* Hidden layer 1: 128,
* Hidden layer 2: 100,
* Output layer: 10 (the number of classes).
<br>For each hidden layer there is:</br>
* ReLU activation function,
* Batch normalization.
<br>The resulting model has hyperparameters as follows:</br>
* Learning rate,
* Dropout for hidden layers,
* Weight decay (L2 regularization),
* Optimizer: Adam or SGD.
<br>The result of building the model part is the buil_model.py script which allows creating, training, and saving the PyTorch deep learning model with the architecture described above. Hyperparameters, as well as the number of epochs for training, may be passed through the command line. The resulting model is loaded and used by the web application described in the next section.
### Building the Web App
The purpose of the web application is to demonstrate how the model can identify the image drawn by the user of the application. The resulting application should let the user:
1.	Draw an image;
2.	Submit the drawing and show the response given by the model.
To solve this problem I used Flask, Bootstrap, Plotly, and PIL library for image processing. The web application consists of following parts:
* Front-end:
    * The main page (index.html, see figure below): The main page allows the user to draw an image with HTML canvas and submit the image. The image is encoded into base64 and passed to Flask server.
    * The results page (hook.html, see figure below): The results page demonstrates the image class identified by the model along with the Plotly diagram, which shows probabilities for each class.
* Back-end (run.py): The Flask server application, which:
    * Loads the deep learning PyTorch model on start;
    * Receives base64 encoded drawing;
    * Preprocesses the drawing;
    * Passes preprocessed drawing to the model;
    * Creates Plotly graph with probabilities for each class;
    * Renders the results page with the message and the graph.
The demo of the application:

![app demo](https://github.com/Lexie88rus/quick-draw-image-recognition/blob/master/assets/demo/demo.gif)
## Repository Structure
The repository has the following structure:
```
- app
| - templates
| |- index.html  # main page of web app
| |- hook.html  # classification result page of web app
| - static
| |- jumbotron.jpg # jumbotron image
|- run.py  # Flask script that runs app

- analysis # folder which contains excel with hyperparameter analysis
- assets # images and calculation results

- build_model.py # script, which builds the classifier to predict customer churn
- build_random_forest.py # script, which builds the classifier using Random Forest
- image_utils.py # script for image manipulations

- README.md
```
## Conclusions
