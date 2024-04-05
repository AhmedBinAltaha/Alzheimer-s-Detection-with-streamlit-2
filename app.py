from __future__ import division, print_function
# coding=utf-8

import os
import numpy as np

# Keras
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
import io
# Flask utils
from flask import Flask, redirect, url_for, request, render_template,jsonify
from werkzeug.utils import secure_filename
from PIL import Image
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model_VGG16.h5'

# Load your trained model
model = load_model(MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5001/')

# Function for processing the input image and prediction
def model_predict(img_path, model):
    # Preprocessing the image
    img = Image.open(io.BytesIO(img_path))
    img = img.resize((224, 224)) 
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
        file_path = secure_filename(f.filename)
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
    
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
    
        image = file.read()

        # Make prediction
        preds = model_predict(image, model)

        # Process your result for human
        class_names = ['AD', 'CN', 'EMCI', 'LMCI']
        pred_class = class_names[preds]

        print("Predicted class:", pred_class)

        # Return the result
        return pred_class
    return None

if __name__ == '__main__':
    app.run(debug=True, port=5001)
