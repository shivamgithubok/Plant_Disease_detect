import os
from PIL import Image
import tensorflow as tf
import numpy as np
import streamlit as st

def load_model():
    model_path = "c:/Users/Asus/Python/Project/Plant/trained_plant_disease_model.keras"
    
    # Check if model exists at the specified path
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please check the path and try again.")
        return None
    
    try:
        # Attempt to load the model
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully.")
        return model
    except OSError as e:
        st.error(f"Error loading model: {e}")
        return None

def model_prediction(test_image):
    model = load_model()
    if model is None:
        return None

    # Ensure the test image is valid and process it
    try:
        image = Image.open(test_image).resize((128, 128))  # Resize for the model input
        input_arr = np.array(image) / 255.0  # Normalize the image
        input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
        predictions = model.predict(input_arr)
        return np.argmax(predictions)  # Return index of max element
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        return None


# Sidebar
st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Display the header image
img = Image.open("Plant.jpg").resize((800, 400))  # Resize to fit page
st.image(img, use_container_width=True)

# Main Page
if app_mode == "HOME":
    st.markdown(
        "<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>",
        unsafe_allow_html=True,
    )

# Prediction Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    test_image = st.file_uploader("Choose an Image (PNG, JPG):", type=["png", "jpg", "jpeg"])

    if test_image is not None:
        # Show the uploaded image
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)

        # Predict button
        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction:")
            result_index = model_prediction(test_image)

            if result_index is not None:
                # Reading Labels
                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy',
                ]
                st.success(f"Model is Predicting it's a {class_name[result_index]}")
            else:
                st.error("Failed to make a prediction. Please check the uploaded image or model file.")
