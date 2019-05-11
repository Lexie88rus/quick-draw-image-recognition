#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May  10 2019

@author: Aleksandra Deis

Script which runs flask web application for Qick Draw

"""
#import libraries
import numpy as np
from PIL import Image
import base64
import re
from io import StringIO
from io import BytesIO
import cv2
import base64
import io
import time

# import Flask
from flask import Flask
from flask import render_template, request

app = Flask(__name__)

# load model

# index webpage receives user input for the model
@app.route('/')
@app.route('/index')
def index():
    # render web page
    return render_template('index.html')

@app.route('/hook', methods=['POST'])
def get_image():
    """
    Get the drawing from the main page.
    """
    # get the base64 string
    image_b64 = request.values['imageBase64']
    image_b64_str = image_b64.split(',')[1].strip()
    # convert string to bytes
    byte_data = base64.b64decode(image_b64_str)
    image_data = BytesIO(byte_data)
    # open Image with PIL
    img = Image.open(image_data)

    # save image as png (for debugging)
    ts = time.time()
    img.save('image' + str(ts) + '.png', 'PNG')

    return ''


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
