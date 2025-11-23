# detector.py
import cv2
import numpy as np
import os

class YOLOv3Detector:
    def __init__(self, model_dir="model", conf_threshold=0.4, nms_threshold=0.3, input_size=416):
        cfg_path = os.path.join(model_dir, "yolov3.cfg")
        weights_path = os.path.join(model_dir, "yolov3.weights")
        names_path = os.path.join(model_dir, "coco.names")

        if not os.path.exists(cfg_path) or not os.path.exists(weights_path) or not os.path.exists(names_path):
            raise FileNotFoundError("One of YOLO model files is missing in 'model/' folder.")

        # Load names
        with open(names_path, "r") as f:
            self.classes = [c.strip() for c in f.readlines()]

        # Load network
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # get output layer names
        self.output_layers = self.net.getUnconnectedOutLayersNames()

        # thresholds
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

    def detect(self, image_bgr):
        """
        image_bgr: image in BGR (OpenCV) format (numpy array)
        returns: list of detections: {"label":str, "confidence":float, "bbox":[x1,y1,x2,y2]}
        """
        H, W = image_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(image_bgr, 1/255.0, (self.input_size, self.input_size), swapRB=True, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        class_ids = []

        # iterate detections
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence > self.conf_threshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # clamp coords
                    x = max(0, x); y = max(0, y)
                    w = int(width); h = int(height)
                    boxes.append([x, y, w, h])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        # apply non-max suppression
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        detections = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, w, h = boxes[i]
                x2 = x + w
                y2 = y + h
                label = self.classes[class_ids[i]] if class_ids[i] < len(self.classes) else str(class_ids[i])
                detections.append({
                    "label": label,
                    "confidence": round(float(confidences[i]), 3),
                    "bbox": [int(x), int(y), int(x2), int(y2)]
                })

        return detections
