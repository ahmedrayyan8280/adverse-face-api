from io import BytesIO
import cv2
from detector import FaceDetectorEngine
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
import numpy as np
from preprocessor import WeatherEnhancer

app = FastAPI(
    title="AdverseFace API",
    description="Real-Time Weather-Robust Face Detection Microservice for Degraded Field Video and Images.",
    version="1.0.0",
)

enhancer = WeatherEnhancer()
detector = FaceDetectorEngine()


async def read_image_from_upload(file: UploadFile) -> np.ndarray:
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Uploaded file must be a valid image."
        )

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400, detail="Corrupted image file received."
        )
    return image


@app.get("/", tags=["Health Check"])
def health_check():
    return {
        "status": "active",
        "service": "AdverseFace API",
        "supported_conditions": ["Rain", "Fog", "Low Contrast"],
    }


@app.post("/api/v1/detect", tags=["Inference"])
async def detect_faces(
    file: UploadFile = File(
        ..., description="Degraded or adverse-weather face image"
    ),
    enhance_weather: bool = Query(
        True,
        description="Apply CLAHE & bilateral filtering prior to detection",
    ),
):
    image = await read_image_from_upload(file)

    if enhance_weather:
        processed_image = enhancer.process(image)
    else:
        processed_image = image

    detection_output = detector.detect(processed_image)

    return JSONResponse(
        content={
            "filename": file.filename,
            "weather_enhancement_applied": enhance_weather,
            "metrics": detection_output,
        }
    )


@app.post("/api/v1/detect/visualize", tags=["Inference"])
async def visualize_detection(
    file: UploadFile = File(...),
    enhance_weather: bool = Query(True),
):
    """Returns the processed image with green bounding boxes drawn over detected faces."""
    image = await read_image_from_upload(file)

    processed_image = (
        enhancer.process(image) if enhance_weather else image.copy()
    )
    detection_output = detector.detect(processed_image)

    # Draw bounding boxes
    for box in detection_output["bounding_boxes"]:
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        cv2.rectangle(
            processed_image, (x, y), (x + w, y + h), (0, 255, 0), 2
        )
        cv2.putText(
            processed_image,
            "Face",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # Encode back to JPEG
    _, encoded_img = cv2.imencode(".jpg", processed_image)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")