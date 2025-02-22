import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image, UnidentifiedImageError
import os

# Load Model with Error Handling
MODEL_PATH = 'best_densenet121_model.h5'
loaded_model = None

if os.path.exists(MODEL_PATH):
    try:
        loaded_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        st.write("✅ Model Loaded Successfully!")
    except Exception as e:
        st.error(f"❌ Error Loading Model: {str(e)}")
else:
    st.error("❌ Model file not found! Check the file path.")

st.title('CIFAR-10 Categories Classification')

# Image Upload Options
genre = st.radio("How You Want To Upload Your Image", ('Browse Photos', 'Camera'))

if genre == 'Camera':
    ImagePath = st.camera_input("Take a picture")
else:
    ImagePath = st.file_uploader("Choose a file", type=['jpeg', 'jpg', 'png'])

# **Ensure the button always appears**
predict_button = st.button('Predict')

# Check if an image is uploaded and model is loaded
if ImagePath is not None and loaded_model is not None:
    try:
        # Open and display image
        image_ = Image.open(ImagePath)
        st.image(image_, width=250, caption="Uploaded Image")

        # Process Image
        loaded_single_image = image_.resize((32, 32))  # Resize to CIFAR-10 model input
        test_image = np.array(loaded_single_image)  # Convert to NumPy array

        # Ensure Image has 3 Channels (RGB)
        if test_image.shape[-1] != 3:
            st.error("❌ Invalid Image Format! Ensure the image has 3 color channels (RGB).")
        else:
            # Normalize and Expand Dimensions
            test_image = test_image / 255.0  # Normalize pixel values
            test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension

            # Debugging: Show Image Shape
            st.write(f"📏 Processed Image Shape: {test_image.shape}")  # Expected: (1, 32, 32, 3)

            # **Run prediction only when button is clicked**
            if predict_button:
                try:
                    logits = loaded_model(test_image)  # Run model prediction

                    # Apply Softmax if the model does not already include it
                    if logits.shape[-1] != 10:  # Ensure logits output matches CIFAR-10 (10 classes)
                        st.error("❌ Model output size mismatch! Expected 10 classes.")
                    else:
                        softmax = tf.nn.softmax(logits)
                        predict_output = tf.argmax(softmax, -1).numpy()[0]

                        # CIFAR-10 class labels
                        classes = ["Airplane", "Automobile", "Bird", "Cat", "Deer",
                                   "Dog", "Frog", "Horse", "Ship", "Truck"]
                        predicted_class = classes[predict_output]
                        probability = softmax.numpy()[0][predict_output] * 100

                        # Display result
                        st.header(f"🟢 Prediction: {predicted_class}")
                        st.subheader(f"🔢 Probability: {probability:.2f}%")

                except Exception as e:
                    st.error(f"❌ Prediction Error: {str(e)}")

    except UnidentifiedImageError:
        st.error('❌ Invalid image format! Please upload a valid JPEG, JPG, or PNG file.')
