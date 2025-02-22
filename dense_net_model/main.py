import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image, UnidentifiedImageError

# Load the trained model
loaded_model = tf.keras.models.load_model("catanddog.h5")

st.title('Cat vs. Dog Classification Using CNN')

# Class labels (adjusted for cat and dog model)
class_names = ['Cat', 'Dog']

# Select image upload method
genre = st.radio(
    "How You Want To Upload Your Image",
    ('Browse Photos', 'Camera'))

if genre == 'Camera':
    ImagePath = st.camera_input("Take a picture")
else:
    ImagePath = st.file_uploader("Choose a file")

# If an image is uploaded
if ImagePath is not None:
    try:
        image_ = Image.open(ImagePath)
        st.image(image_, width=250)

        if st.button('Predict'):
            # Load image and preprocess
            test_image = image.load_img(ImagePath, target_size=(224, 224))  # Match model input size
            test_image = image.img_to_array(test_image)
            test_image /= 255.0  # Normalize pixel values
            test_image = np.expand_dims(test_image, axis=0)

            # Make prediction
            logits = loaded_model.predict(test_image, verbose=0)
            softmax = tf.nn.softmax(logits)
            predict_output = np.argmax(logits, axis=-1)[0]

            predicted_class = class_names[predict_output]
            probability = softmax.numpy()[0][predict_output] * 100

            # Display results
            st.header(f'Prediction: {predicted_class}')
            st.header(f'Confidence: {probability:.2f} %')

    except UnidentifiedImageError:
        st.write('Input Valid File Format !!!  [jpeg, jpg, png only this format is supported !]')

except TypeError:
    st.header('Please Upload Your File !!!')

except UnidentifiedImageError:
    st.header('Input Valid File !!!')
