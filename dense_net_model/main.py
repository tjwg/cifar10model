import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image, UnidentifiedImageError
import os

def set_background(image_url):
    """
    Sets a background image for the Streamlit app.
    """
    background_style = f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
    """
    st.markdown(
    """
    <style>
    .stApp {
        background: url("https://upload.wikimedia.org/wikipedia/commons/f/f3/Rembrandt_Christ_in_the_Storm_on_the_Lake_of_Galilee.jpg") no-repeat center center fixed;
        background-size: cover;
        position: relative;
    }
    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.4); /* Dark overlay with 40% opacity */
        z-index: 0;
    }
    h1, h2, h3, h4, h5, h6, p, .stMarkdown {
        color: white !important; /* Ensures text is readable */
        position: relative;
        z-index: 1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model safely
MODEL_PATH = 'dense_net_model/best_densenet121_model.h5'

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
            logits = loaded_model.predict(test_image)
            softmax = tf.nn.softmax(logits)
            predict_output = tf.argmax(logits, -1).numpy()[0]

            # CIFAR-10 class labels
            classes = ["Airplane", "Automobile", "Bird", "Cat", "Deer",
                       "Dog", "Frog", "Horse", "Ship", "Truck"]
            predicted_class = classes[predict_output]
            probability = softmax.numpy()[0][predict_output] * 370

            # Display result
            st.header(f"Prediction: {predicted_class}")
            st.subheader(f"Probability: {probability:.2f}%")

    except UnidentifiedImageError:
        st.error('Invalid image format! Please upload a valid JPEG, JPG, or PNG file.')
