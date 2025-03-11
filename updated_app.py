import streamlit as st
import numpy as np
import cvlib as cv
import tensorflow as tf
import pandas as pd
import json
import av
from datetime import datetime
import os

from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from PIL import Image, ImageDraw, ImageFont
from streamlit_lottie import st_lottie
from deepface import DeepFace

# Streamlit Page Configurations
st.set_page_config(layout='wide', page_title="Face Vision", page_icon="👦")
con1 = st.container()
con2 = st.container()

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load Lottie animation
logo = load_lottiefile("animation.json")

def web_emotion_detection(frame):
    image_np = frame.to_ndarray(format="bgr24")
    image_pil = Image.fromarray(image_np)
    draw = ImageDraw.Draw(image_pil)
    faces, confidences = cv.detect_face(image_np)
    for idx, f in enumerate(faces):
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
        face_img = image_np[startY:endY, startX:endX]
        obj = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
        emotions = (d["dominant_emotion"] for d in obj)
        draw.rectangle(((startX, startY), (endX, endY)), outline=(0, 255, 0), width=2)
        for i, emotion in enumerate(emotions):
            label = emotion
            font = ImageFont.truetype("arial.ttf", 15)
            draw.text((startX+10, startY-20), label, font=font, fill=(0, 255, 0))
    image_np = np.array(image_pil)
    return av.VideoFrame.from_ndarray(image_np, format="bgr24")

# Sidebar
with st.sidebar:
    st.title('Experience the power of computer vision')
    st.success('This is a computer vision application that detects emotions in real-time using the webcam')
    st.info('This application uses the DeepFace library to detect emotions in real-time')
    task2 = ['<Select>', 'Capture', 'Web-Cam']
    mode = st.selectbox('Select Mode', task2)

# Main UI
with con1:
    st.markdown(
        """
        <style>
            .title {
                text-align: center;
                font-weight: bold;
            }
            .mainheading {
                text-align: center;
                font-family: monospace;
                font-size: 25px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<h1 class="title">Face Vision</h1>', unsafe_allow_html=True)
    st.markdown('<br><br>', unsafe_allow_html=True)
    
with con2:
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        c_1, c_2, c_3 = st.columns([1, 3, 1])
        with c_2:
            st_lottie(logo, speed=1, width=250, height=250)
        st.markdown('<br><br>', unsafe_allow_html=True)
        if mode == '<Select>':
            st.warning('👈 Explore the power of computer vision by selecting a mode from the sidebar')

# Directory for saving images
output_dir = "detected_images"
os.makedirs(output_dir, exist_ok=True)

# Capture Mode
if mode == 'Capture':
    c_1, c_2, c_3 = st.columns([1, 3, 1])
    with c_2:
        image = st.camera_input("Pick a snapshot")
        if image is not None:
            image = Image.open(image)
            image_np = np.array(image)
            image_pil = Image.fromarray(image_np)
            draw = ImageDraw.Draw(image_pil)
            faces, confidences = cv.detect_face(image_np)
            for f in faces:
                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]
                draw.rectangle(((startX, startY), (endX, endY)), outline=(0, 255, 0), width=2)
                face_img = image_np[startY:endY, startX:endX]
                obj = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                emotions = (d["dominant_emotion"] for d in obj)
                for i, emotion in enumerate(emotions):
                    label = emotion
                    font = ImageFont.truetype("arial.ttf", 15)
                    draw.text((startX+10, startY-20), label, font=font, fill=(0, 255, 0))
            image_np = np.array(image_pil)
            st.image(image_np, caption='Uploaded Image.', use_column_width=True)

            # Save the processed image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(output_dir, f"detected_image_{timestamp}.jpg")
            image_pil.save(save_path)
            st.success(f"Image saved to {save_path}")

if mode == 'Web-Cam':
    c_1, c_2, c_3 = st.columns([1,3,1])
    with c_2:
        webrtc_streamer(key="example", video_frame_callback=web_emotion_detection,
    )