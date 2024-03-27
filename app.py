from __future__ import division, print_function
# coding=utf-8
import json
import sys
import os
import glob
import re
import numpy as np
from flask import  jsonify

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'new_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
# print('Model loaded. Start serving...')

print('Model loaded. Check http://127.0.0.1:5000/')


#function for processing the input image abd prediction
def model_predict(img_path, model):
    # Preprocessing the image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x/255.0

    print("Input image shape:", x.shape)
    print("Input image values:", x)

    y = model.predict(x)
    print("Model predictions:", y)

    pred_class_index = np.argmax(y)
    print("Predicted class index:", pred_class_index)

    return pred_class_index

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        
        # Process your result for human
        class_names = ['AD', 'CN', 'EMCI', 'LMCI']
        pred_class = class_names[preds]

        print("Predicted class:", pred_class)

        # Return the result
        return pred_class
    
               
    return None



if __name__ == '__main__':
    app.run(debug=True, port=5000)

