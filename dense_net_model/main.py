import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image, UnidentifiedImageError
import os

# Load model safely
MODEL_PATH = 'best_densenet_model.h5'
loaded_model = tf.keras.models.load_model(MODEL_PATH)
#loaded_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
#loaded_model.build((None, 32, 32, 3))  # Adjust this if your model expects a different size

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found! Check the file path.")
else:
    loaded_model = tf.keras.models.load_model(MODEL_PATH)

st.title('CIFAR-10 Categories Classification')

# Image upload options
genre = st.radio("How You Want To Upload Your Image", ('Browse Photos', 'Camera'))

if genre == 'Camera':
    ImagePath = st.camera_input("Take a picture")
else:
    ImagePath = st.file_uploader("Choose a file", type=['jpeg', 'jpg', 'png'])

if ImagePath is not None:
    try:
        # Open image with PIL
        image_ = Image.open(ImagePath)
        st.image(image_, width=250, caption="Uploaded Image")

        # Process and predict when button is clicked
        if st.button('Predict'):
            loaded_single_image = image_.resize((32, 32))  # Resize for model input
            test_image = np.array(loaded_single_image) / 255.0  # Normalize
            test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension

            # Model prediction
            logits = loaded_model(test_image)
            softmax = tf.nn.softmax(logits)
            predict_output = tf.argmax(logits, -1).numpy()[0]

            # CIFAR-10 class labels
            classes = ["Airplane", "Automobile", "Bird", "Cat", "Deer",
                       "Dog", "Frog", "Horse", "Ship", "Truck"]
            predicted_class = classes[predict_output]
            probability = softmax.numpy()[0][predict_output] * 100

            # Display result
            st.header(f"Prediction: {predicted_class}")
            st.subheader(f"Probability: {probability:.2f}%")

    except UnidentifiedImageError:
        st.error('Invalid image format! Please upload a valid JPEG, JPG, or PNG file.')
