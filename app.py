import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load Haar cascade classifier
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

st.title("📸 Haar Cascade Face Detection")
st.write("Upload an image and detect faces using OpenCV Haar Cascade Classifier.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image with PIL
    image = Image.open(uploaded_file).convert("RGB")  # Ensure 3-channel RGB
    img_array = np.array(image)

    # Convert RGB to BGR (for OpenCV)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = haar_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw bounding boxes
    for (x, y, w, h) in faces:
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert back to RGB for Streamlit
    result_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Display result
    st.image(result_img, caption=f"Detected {len(faces)} face(s)", use_container_width=True)

    st.success(f"✅ Found {len(faces)} face(s) in the uploaded image.")
