# Microservice YOLO Object Detection Pipeline

This project implements a complete microservice-based Object Detection system using the YOLOv3 model. It includes a FastAPI backend for AI inference and a Streamlit-based frontend for user interaction. Users can upload any image, run object detection, and instantly receive an annotated output. The objective of this project is to provide a production-ready and scalable object detection service that can be deployed on local machines, servers, Docker containers, and cloud platforms like AWS or GCP.


The model supports 80+ COCO classes like: Person, Car, Dog, Bicycle, Bus, Bottle, Cat, etc.

## Backend (FastAPI)
cd AI_Backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8001

API Docs → http://localhost:8001/docs

## Frontend (Streamlit)
cd UI_Backend
streamlit run app.py


## Testing Suggestions

Upload images with:
- Cars & People on the street
- Dog / Cat images
- Good lighting & clear object visibility
- COCO dataset sample images for best detection


## Technologies Used

| Component | Tech |
|----------|------|
| AI Model | YOLOv3 (OpenCV DNN) |
| Backend | FastAPI |
| Frontend | Streamlit |
| Image Ops | Pillow, OpenCV, NumPy |
| Language | Python 3.10+ |

