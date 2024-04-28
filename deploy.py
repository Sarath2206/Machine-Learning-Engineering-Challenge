import streamlit as st
import numpy as np
import pickle
from PIL import Image
import tensorflow as tf

# Function to load the CNN model from .pkl file
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((32, 32))  # Resize image to match CIFAR-10 dataset
    image = np.array(image)
    # Normalize pixel values
    image = (image - 127.5) / 127.5
    return image

# Function to classify the image
def classify_image(image, model):
    image = preprocess_image(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    return class_index

# Main function to create the Streamlit web application
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Home", "Predict"])

    if page == "Home":
        st.title("Welcome to CIFAR-10 Image Classifier")
        st.write("Please select the 'Predict' page from the sidebar to classify an image.")
    elif page == "Predict":
        st.title("CIFAR-10 Image Classifier")

        # Load pre-trained model
        model_path = "IISC\model-CIFAR10.pkl"  # Replace with your model path
        model = load_model(model_path)

        # Upload image
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.write("")
            st.write("Classifying...")
            class_index = classify_image(image, model)

            # Display prediction
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            st.write("Predicted class:", class_names[class_index])

if __name__ == "__main__":
    main()