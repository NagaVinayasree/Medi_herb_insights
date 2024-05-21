import streamlit as st
import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

# Load the best model
best_model = load_model('best_model.h5')

# Define the image size used during training
IMAGE_SIZE = (299, 299)

# Function to preprocess an image for prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Function to perform prediction on a given image
def predict_image(model, img_path):
    preprocessed_img = preprocess_image(img_path)
    prediction = model.predict(preprocessed_img)
    return prediction

# Function to display the prediction result
def display_prediction(prediction, class_labels):
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_labels[predicted_class_index]
    confidence = prediction[0][predicted_class_index]

    st.write("Predicted Class:", predicted_class_label)
    st.write("Confidence:", f"{confidence * 100:.2f}%")

# Streamlit App
def main():
    st.title("MediHerb Insight")
    st.sidebar.header("Upload Image")

    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Load class labels from the dataset directory
        data_dir = 'C:/Users/sneha/OneDrive/Desktop/project/model/Medicinal Leaf dataset'
        class_labels = sorted(os.listdir(data_dir))

        # Make a prediction
        prediction_result = predict_image(best_model, uploaded_file)

        # Display the prediction
        display_prediction(prediction_result, class_labels)

if __name__ == "__main__":
    main()