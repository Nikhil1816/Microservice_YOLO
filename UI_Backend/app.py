import streamlit as st
import requests
from PIL import Image
import io
import os
from dotenv import load_dotenv
from utils import ensure_dir
import base64

load_dotenv()

AI_URL = os.getenv("AI_URL", "http://localhost:8001/detect")  
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "../output")
ensure_dir(OUTPUT_DIR)

st.set_page_config(page_title="Object Detection UI", layout="wide")

st.title("Object Detection — Streamlit UI")
st.write("Upload an image → sent to AI backend → shows AI output (annotated image + JSON)")

with st.sidebar:
    st.header("Configuration")
    st.write("AI Endpoint:")
    st.code(AI_URL)
    st.write("Output folder:")
    st.code(OUTPUT_DIR)

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Please upload an image to start detection.")
    st.stop()

image_bytes = uploaded_file.read()
try:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
except Exception as e:
    st.error(f"Error reading image: {e}")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Image")
    st.image(image, use_column_width=True)

if st.button("Run Detection"):
    status = st.empty()
    status.info("Sending image to AI backend...")

    try:
        files = {"file": (uploaded_file.name, image_bytes, uploaded_file.type)}
        response = requests.post(AI_URL, files=files, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        status.error(f"AI backend error: {e}")
        st.stop()

    status.success("Detection complete!")

    annotated_bytes = base64.b64decode(data["annotated_image"])
    annotated_img = Image.open(io.BytesIO(annotated_bytes))

    with col2:
        st.subheader("Detected Output")
        st.image(annotated_img, use_column_width=True)
        st.subheader("Detections JSON")
        st.json(data.get("detections", []))

    with st.expander("Download results"):
        st.download_button(
            "Download Annotated Image",
            data=annotated_bytes,
            file_name="detected_output.jpg",
            mime="image/jpeg"
        )

        st.download_button(
            "Download JSON",
            data=io.BytesIO(response.content),
            file_name="detections.json",
            mime="application/json"
        )
