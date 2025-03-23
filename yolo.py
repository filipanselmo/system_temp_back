import cv2
import torch
import json


class YOLO:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Поменять на свою модель, обученную на датасете

    def detect_objects(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Image not found or could not be read.")
            # Возвращаем пустой JSON
            return json.dumps([])
        results = self.model(img)
        print("Raw detection results:", results)
        # Возвращает результаты в формате JSON
        return results.pandas().xyxy[0].to_json(orient="records")
