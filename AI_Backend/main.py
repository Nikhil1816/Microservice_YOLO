from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import base64
from detector import YOLOv3Detector
from utils.draw import draw_boxes
from utils.image_utils import bytes_to_cv2, cv2_to_pil, pil_to_bytes
import uvicorn

app = FastAPI(title="AI Backend - YOLOv3 Detector")

detector = YOLOv3Detector(model_dir="model", conf_threshold=0.4, nms_threshold=0.3, input_size=416)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()
    image_bgr = bytes_to_cv2(image_bytes)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Unable to decode image.")

    detections = detector.detect(image_bgr)

    pil_img = cv2_to_pil(image_bgr)
    annotated = draw_boxes(pil_img, detections)

    annotated_bytes = pil_to_bytes(annotated, fmt="JPEG", quality=85)
    annotated_b64 = base64.b64encode(annotated_bytes).decode("utf-8")

    return JSONResponse({
        "detections": detections,
        "annotated_image": annotated_b64
    })

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
