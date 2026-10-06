import time
from typing import Any, Dict, List
import cv2
import mediapipe as mp
import numpy as np


class FaceDetectorEngine:

    def __init__(self, min_detection_confidence: float = 0.35):
        # MediaPipe BlazeFace: model_selection=1 optimizes for full-range/field shots
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        start_time = time.perf_counter()
        img_height, img_width, _ = image.shape

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_image)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        bounding_boxes: List[Dict[str, Any]] = []

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                score = round(float(detection.score[0]), 3)

                x = int(bbox.xmin * img_width)
                y = int(bbox.ymin * img_height)
                w = int(bbox.width * img_width)
                h = int(bbox.height * img_height)

                x = max(0, x)
                y = max(0, y)

                bounding_boxes.append(
                    {
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "confidence": score,
                    }
                )

        return {
            "total_faces_detected": len(bounding_boxes),
            "inference_latency_ms": latency_ms,
            "bounding_boxes": bounding_boxes,
        }