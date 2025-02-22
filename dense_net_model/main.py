import streamlit as st
from tensorflow.keras.models import Sequential, model_from_json
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import UnidentifiedImageError
from PIL import Image

loaded_model = tf.keras.models.load_model('best_densenet121_model.h5')

try:
    if st.button('Predict'):
        loaded_single_image = tf.keras.utils.load_img(ImagePath,
                                                      color_mode='rgb',
                                                      target_size=(32, 32))  # edit to model input size

        test_image = tf.keras.utils.img_to_array(loaded_single_image)
        test_image /= 255

        test_image = np.expand_dims(test_image, axis=0)

        logits = loaded_model(test_image)
        softmax = tf.nn.softmax(logits)

        predict_output = tf.argmax(logits, -1).numpy()
        classes = ["Airplane", "Automobile", "Bird", "Cat", "Deer",
               "Dog", "Frog", "Horse", "Ship", "Truck"]
        st.header(f"Prediction: {classes[predict_output[0]]}")

        predicted_class = classes[predict_output[0]]

        # Get the probability of the predicted class
        probability = softmax.numpy()[0][predict_output[0]] * 100

        # probability = predict_output[0][predicted_class_index] * 100
        st.header(f"Probability of a {predicted_class}: {probability:.4f}%")
