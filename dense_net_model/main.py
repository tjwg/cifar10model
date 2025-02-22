import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image, UnidentifiedImageError
import os

# Load model safely
MODEL_PATH = 'best_densenet_model.h5'

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found! Please upload or check the model path.")
    st.stop()  # Stops execution if the model is missing

loaded_model = tf.keras.models.load_model(MODEL_PATH)

st.title('CIFAR-10 Image Classification')

# CIFAR-10 class labels
classes = ["Airplane", "Automobile", "Bird", "Cat", "Deer",
           "Dog", "Frog", "Horse", "Ship", "Truck"]

# Image upload options
genre = st.radio("How would you like to upload your image?", ('Browse Photos', 'Camera'))

if genre == 'Camera':
    ImagePath = st.camera_input("Take a picture")
else:
    ImagePath = st.file_uploader("Choose a file", type=['jpeg', 'jpg', 'png'])

if ImagePath is not None:
    try:
        # Open and display the uploaded image
        image_ = Image.open(ImagePath)
        st.image(image_, caption="Uploaded Image", width=250)

        # Predict button
        if st.button('Predict'):
            # Convert and preprocess the image
            image_resized = image_.resize((32, 32))  # Resize to match model input
            test_image = np.array(image_resized) / 255.0  # Normalize pixel values
            test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension

            # Make prediction
            logits = loaded_model.predict(test_image, verbose=0)
            softmax = tf.nn.softmax(logits)
            predict_output = np.argmax(logits, axis=-1)[0]

            predicted_class = classes[predict_output]
            probability = softmax.numpy()[0][predict_output] * 100

            # Display predictions
            st.header(f"Prediction: {predicted_class}")
            st.subheader(f"Confidence: {probability:.2f}%")

    except UnidentifiedImageError:
        st.error("Invalid image format! Please upload a valid JPEG, JPG, or PNG file.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
