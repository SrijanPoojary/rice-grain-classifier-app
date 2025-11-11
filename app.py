import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image 
import time
import io

# --- 0. Configuration and Initial Setup ---
MODEL_PATH = 'rice_classifier_mobilenetv2.h5'
IMG_HEIGHT = 224
IMG_WIDTH = 224

# The class names must be in the same order as they were during training
class_names = ['Arborio', 'Basmati', 'Gonen', 'Ipsala', 'Jasmine']

# --- Load the Trained Model (with Streamlit caching for efficiency) ---
@st.cache_resource 
def load_my_model():
    """Loads the pre-trained Keras model."""
    try:
        # Load the model from the specified path
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        # Display error if model loading fails
        st.error(f"Error loading model: Could not find or load '{MODEL_PATH}'. Ensure the file is in the correct directory. Details: {e}")
        return None

model = load_my_model()

# --- Streamlit Page Configuration ---
# Setting the page to wide layout for a professional, spacious look
st.set_page_config(
    layout="wide", 
    page_title=" Rice Grain Classifier", 
    page_icon="🍚"
)

# --- 1. Sidebar for Project Information and Instructions ---
st.sidebar.title("📚 Project Details")
st.sidebar.info(
    """
   This Deep Learning project implements an automated solution for rice grain
    classification. The core of the system is a MobileNetV2 model, which has been 
    trained and validated on five distinct rice varieties (Arborio, Basmati, Gonen, 
    Ipsala, and Jasmine) to ensure accurate and reliable identification.
    """
)
st.sidebar.markdown("---")
st.sidebar.subheader("Instructions")
st.sidebar.markdown("1. Upload a high-resolution image of the rice sample in the main section.")
st.sidebar.markdown("2. Click the **Classify Image** button.")
st.sidebar.markdown("3. View the prediction and confidence score below.")


# --- 2. Main Title and Header ---
st.title("  Rice Grain Classification ")
st.markdown("A rapid and accurate system for variety identification.")
st.markdown("---")

if model is None:
    # Stop the application display if the model failed to load
    st.stop() 

# --- 3. Column Layout for Input and Results ---
col_input, col_output = st.columns([1, 2]) # 1/3 width for input, 2/3 for output

# --- Input Column ---
with col_input:
    st.subheader("1. Upload Rice Image")
    
    uploaded_file = st.file_uploader(
        "Select a file to classify:", 
        type=["jpg", "jpeg", "png"]
    )
    
    # Create a placeholder for the image to be displayed
    img_placeholder = st.empty()


# --- Output/Classification Column ---
with col_output:
    
    if uploaded_file is not None:
        
        # Display the uploaded image immediately
        img = Image.open(uploaded_file).convert('RGB')
        # FIX: Replaced deprecated 'use_column_width' with 'use_container_width'
        img_placeholder.image(img, caption='Uploaded Rice Sample.', use_container_width=True) 
        
        st.markdown("---")
        
        # 4. Classification Button
        if st.button('🚀 Classify Image', type="primary", use_container_width=True):
            
            # 5. Progress Indicator (Spinner)
            with st.spinner('⏳ Running MobileNetV2 model and calculating prediction...'):
                
                # --- START: UNTOUCHED MODEL LOGIC ---
                # Preprocess the image for the model
                img_resized = img.resize((IMG_HEIGHT, IMG_WIDTH))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) # Create a batch dimension
                img_array /= 255.0 # Rescale pixels to [0, 1]

                # Make prediction
                try:
                    predictions = model.predict(img_array)
                    predicted_class_index = np.argmax(predictions[0])
                    predicted_class_name = class_names[predicted_class_index]
                    confidence = np.max(predictions[0]) * 100

                    # Add a small delay for a cleaner visual effect
                    time.sleep(1) 

                    st.success('✅ Classification Complete!')
                    st.markdown("---")

                    # --- 6. Display Results with Metrics ---
                    st.subheader("2. Final Prediction Summary")
                    
                    # Use columns for neat metric display
                    col_res1, col_res2 = st.columns(2)
                    
                    col_res1.metric(
                        label="Predicted Rice Variety", 
                        value=predicted_class_name,
                        delta=f"{confidence:.2f}% Confidence"
                    )
                    
                    col_res2.metric(
                        label="Probability Score", 
                        value=f"{confidence:.2f}%"
                    )

                    # Optional: Display all probabilities in an expander for details
                    st.markdown("---")
                    with st.expander("Detailed Class Probabilities", expanded=False):
                        st.table(
                            data=[
                                {"Variety": name, "Probability": f"{prob*100:.2f}%"}
                                for name, prob in zip(class_names, predictions[0])
                            ]
                        )

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                # --- END: UNTOUCHED MODEL LOGIC ---

    else:
        st.info("Upload an image on the left to activate the classifier.")


st.markdown("---")
st.caption("Developed using TensorFlow and Streamlit for a fast, user-friendly submission.")
