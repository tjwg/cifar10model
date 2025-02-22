import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image, UnidentifiedImageError

# Load the pre-trained CIFAR-10 model
loaded_model = tf.keras.models.load_model('best_densenet_model.h5')

st.title('CIFAR-10 Image Classification')

# CIFAR-10 class labels
classes = ["Airplane", "Automobile", "Bird", "Cat", "Deer",
           "Dog", "Frog", "Horse", "Ship", "Truck"]

# Select image upload method
genre = st.radio(
    "How would you like to upload your image?",
    ('Browse Photos', 'Camera'))

if genre == 'Camera':
    ImagePath = st.camera_input("Take a picture")
else:
    ImagePath = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

if ImagePath is not None:
    try:
        image_ = Image.open(ImagePath)
        st.image(image_, caption="Uploaded Image", width=250)

        if st.button('Predict'):
            # Load and preprocess the image
            loaded_single_image = image.load_img(ImagePath, target_size=(32, 32))  # CIFAR-10 model input size
            test_image = image.img_to_array(loaded_single_image)
            test_image = test_image / 255.0  # Normalize pixel values
            test_image = np.expand_dims(test_image, axis=0)

            # Get model predictions
            logits = loaded_model.predict(test_image, verbose=0)
            softmax = tf.nn.softmax(logits)
            predict_output = np.argmax(logits, axis=-1)[0]

            predicted_class = classes[predict_output]
            probability = softmax.numpy()[0][predict_output] * 100

            # Display predictions
            st.header(f"Prediction: {predicted_class}")
            st.subheader(f"Confidence: {probability:.2f}%")

    except UnidentifiedImageError:
        st.error("Invalid file format! Please upload a valid JPEG, JPG, or PNG file.")

except TypeError:
    st.error('Please upload an image first!')
