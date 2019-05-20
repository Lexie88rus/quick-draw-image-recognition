# Quick, Draw! Image Recognition
Recognition of Quick, Draw! game doodles.

## DEFINITION
### Project Overview
[The Quick Draw Dataset](https://github.com/googlecreativelab/quickdraw-dataset) is a collection of 50 million drawings across 345 categories, contributed by players of the game Quick, Draw!. The player starts with an object to draw (for example it may say "Draw a chair in under 20 seconds"). Then the player has twenty seconds to draw that object. Based on what they draw, the AI guesses what they are drawing.
Research in recognition of images drawn by humans can improve pattern recognition solutions more broadly. Improving pattern recognition has an impact on handwriting recognition and its robust applications in areas including OCR (Optical Character Recognition), ASR (Automatic Speech Recognition) & NLP (Natural Language Processing).
In this project I analyzed the drawings and tried to build a deep learning application to classify those drawings.
### Problem Statement
Recognition of a drawing is a classification problem. I have to build a solution, which classifies input images. I split the whole problem of recognition of drawings into the following tasks:
•	Input data analysis and preprocessing;
•	Building a deep learning model to classify drawings;
•	Evaluation of the model concerning chosen metrics;
•	Building a web-application to demonstrate the results.
### Metrics
I chose accuracy as a metric to evaluate the results. Because of the rules of the game, we mostly care about how many times did the AI recognize the drawing correctly, and this is just the accuracy of the model.
