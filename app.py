import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import torch

from models import get_classification_model, get_detection_model, get_classes, get_coco_names
from utils import preprocess_image, draw_bounding_boxes

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Image Classifier & Object Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("static/style.css")

# --- App Title and Description ---
st.title("AI Image Classifier & Object Detector")
st.markdown(
    """
    <p class="big-font">A versatile web app for real-time image classification and object detection.</p>
    <p class="medium-font">Upload an image or use your webcam to get started.</p>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Options")
    app_mode = st.selectbox("Choose the App Mode", ["Image Classification", "Object Detection"])
    st.markdown("---")
    
    if app_mode == "Image Classification":
        st.subheader("Classification Settings")
        top_k = st.slider("Top-K Predictions", 1, 10, 5)
    
    if app_mode == "Object Detection":
        st.subheader("Detection Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

    st.markdown("---")
    st.subheader("Image Source")
    image_source = st.radio("Select Image Source", ["Upload an Image", "Use Webcam"])

# --- Image Input ---
image_file = None
if image_source == "Upload an Image":
    image_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])
elif image_source == "Use Webcam":
    st.info("Webcam functionality is not yet implemented.")
    # Placeholder for webcam integration
    # webcam_image = st.camera_input("Take a picture")
    # if webcam_image:
    #     image_file = webcam_image

# --- Main Content ---
if image_file is not None:
    image = Image.open(image_file).convert("RGB")
    
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.markdown("<h3 class='result-header'>Results</h3>", unsafe_allow_html=True)
        
        with st.spinner("Analyzing the image..."):
            start_time = time.time()

            if app_mode == "Image Classification":
                # --- Classification Logic ---
                model = get_classification_model()
                classes = get_classes()
                
                processed_image = preprocess_image(image)
                with torch.no_grad():
                    outputs = model(processed_image)
                
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                top_k_probs, top_k_indices = torch.topk(probabilities, top_k)

                st.markdown("#### Top Predictions:")
                for i in range(top_k):
                    class_name = classes[top_k_indices[i]]
                    probability = top_k_probs[i].item()
                    st.write(f"<p class='prediction'>{i+1}. {class_name}: {probability:.2%}</p>", unsafe_allow_html=True)

            elif app_mode == "Object Detection":
                # --- Detection Logic ---
                model = get_detection_model()
                coco_names = get_coco_names()
                
                img_tensor = preprocess_image(image, detection=True)
                
                with torch.no_grad():
                    predictions = model(img_tensor)

                img_with_boxes = draw_bounding_boxes(image, predictions, coco_names, confidence_threshold)
                st.image(img_with_boxes, caption="Image with Detected Objects", use_column_width=True)

            end_time = time.time()
            inference_time = end_time - start_time
            st.markdown(f"<p class='inference-time'>Inference Time: {inference_time:.4f} seconds</p>", unsafe_allow_html=True)

else:
    st.info("Please upload an image or select the webcam option to proceed.")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <p class="footer">Developed with ❤️ by Gemini</p>
    """,
    unsafe_allow_html=True,
)
