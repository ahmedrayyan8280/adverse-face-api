import cv2
from detector import FaceDetectorEngine
import numpy as np
from PIL import Image
from preprocessor import WeatherEnhancer
import streamlit as st

st.set_page_config(page_title="AdverseFace Vision UI", layout="wide")

st.title("🌧️ AdverseFace: Weather-Robust Face Detection")
st.write(
    "A deep learning vision pipeline engineered to detect human faces under severe rain, fog, and contrast degradation."
)

# Initialize engines
enhancer = WeatherEnhancer()
detector = FaceDetectorEngine(min_detection_confidence=0.35)

# Sidebar configurations
st.sidebar.header("Configuration & Telemetry")
apply_enhancement = st.sidebar.checkbox(
    "Apply Weather Filtering (CLAHE + Bilateral)", value=True
)
draw_confidence = st.sidebar.checkbox("Show Confidence Scores", value=True)

uploaded_file = st.file_uploader(
    "Upload surveillance / field photo...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Decode image buffer into OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Execute processing pipeline
    processed_image = (
        enhancer.process(image) if apply_enhancement else image.copy()
    )
    results = detector.detect(processed_image)

    # Annotate bounding boxes
    annotated_image = processed_image.copy()
    for box in results["bounding_boxes"]:
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        conf = box["confidence"]
        label = f"Face {int(conf * 100)}%" if draw_confidence else "Face"

        cv2.rectangle(
            annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 3
        )
        cv2.putText(
            annotated_image,
            label,
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # Side-by-side comparison
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw Input")
        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True
        )

    with col2:
        st.subheader("Enhanced & Detected Output")
        st.image(
            cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )

    # Telemetry Card
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Faces Detected", results["total_faces_detected"])
    m2.metric("Inference Latency", f"{results['inference_latency_ms']} ms")
    m3.metric(
        "Preprocessing Active",
        "CLAHE + Bilateral" if apply_enhancement else "Bypassed",
    )