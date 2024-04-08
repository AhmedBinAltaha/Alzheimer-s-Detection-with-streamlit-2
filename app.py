import io
import streamlit as st
from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image

# Load your trained model
MODEL_PATH = 'model_VGG16.h5'
model = load_model(MODEL_PATH)

# Initialize Flask app
app = Flask(__name__)

# Function for processing the input image and prediction
def model_predict(img_data, model):
    # Preprocessing the image
    img = img_data.resize((224, 224))  # Resize the image directly
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    print("Input image shape:", x.shape)
    print("Input image values:", x)

    y = model.predict(x)
    print("Model predictions:", y)

    pred_class_index = np.argmax(y)
    print("Predicted class index:", pred_class_index)

    return pred_class_index


# Main Streamlit app
def main():
    st.title("Alzheimer's Disease Diagnosis App")
    st.write("The Site For Alzheimer's Disease Diagnosis For 4 Class ADNI Data Categories")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=False, width=224)

        st.write("")
        st.write("Classifying...")

        # Open the uploaded image
        img = Image.open(uploaded_file)

        # Make prediction
        preds = model_predict(img, model)  # Here, pass the Image object directly

        # Process your result for human
        class_names = ['AD', 'CN', 'EMCI', 'LMCI']
        pred_class = class_names[preds]

        st.success(f"The predicted class is: {pred_class}")

# Define API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file)
    preds = model_predict(img, model)
    class_names = ['AD', 'CN', 'EMCI', 'LMCI']
    pred_class = class_names[preds]
    return jsonify({'predicted_class': pred_class})

# Run the Flask app
if __name__ == '__main__':
    main()
    app.run(port=5001)
