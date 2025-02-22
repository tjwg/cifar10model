import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError

# Load the pre-trained DenseNet121 model
loaded_model = tf.keras.models.load_model('best_densenet121_model.h5')

# Radio button for image upload options
genre = st.radio("How You Want To Upload Your Image", ('Browse Photos', 'Camera'))

if genre == 'Camera':
    ImagePath = st.camera_input("Take a picture")
else:
    ImagePath = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

# Ensure an image is uploaded or taken
if ImagePath is not None:
    try:
        # Open the image using PIL
        image_ = Image.open(ImagePath)

        # Display the uploaded image
        st.image(image_, width=250)

        # Convert the image to the correct format for model input
        loaded_single_image = image_.convert("RGB")  # Ensure it's RGB
        loaded_single_image = loaded_single_image.resize((32, 32))  # Resize to model's expected size
        test_image = np.array(loaded_single_image) / 255.0  # Normalize pixel values
        test_image = np.expand_dims(test_image, axis=0)  # Expand dimensions for batch input

        # Predict button
        if st.button('Predict'):
            try:
                logits = loaded_model(test_image)  # Get model logits
                softmax = tf.nn.softmax(logits)  # Convert logits to probabilities

                predict_output = tf.argmax(softmax, -1).numpy()  # Get predicted class index
                classes = ["Airplane", "Automobile", "Bird", "Cat", "Deer",
                           "Dog", "Frog", "Horse", "Ship", "Truck"]
                predicted_class = classes[predict_output[0]]

                # Get the probability of the predicted class
                probability = softmax.numpy()[0][predict_output[0]] * 100

                # Display results
                st.header(f"Prediction: {predicted_class}")
                st.header(f"Probability of a {predicted_class}: {probability:.4f}%")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

    except UnidentifiedImageError:
        st.error("Invalid file format! Please upload an image in JPEG, JPG, or PNG format.")

    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")

