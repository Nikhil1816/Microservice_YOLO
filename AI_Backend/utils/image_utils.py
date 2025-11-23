# utils/image_utils.py
import cv2
import numpy as np
from PIL import Image
import io

def bytes_to_cv2(image_bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def cv2_to_pil(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)

def pil_to_bytes(pil_img, fmt="JPEG", quality=90):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt, quality=quality)
    buf.seek(0)
    return buf.getvalue()
