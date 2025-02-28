import cv2
from fastapi import FastAPI
from starlette.responses import StreamingResponse
from ultralytics import YOLO

app = FastAPI()

# Load the YOLOv8 model from the file model.pt
model = YOLO("model.pt")


def generate_raw_frames():
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
    while True:
        success, frame = cap.read()
        if not success:
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )


def generate_prediction_frames():
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run the YOLOv8 model on the frame
        results = model(frame)
        # Get the annotated frame with predictions overlaid
        annotated_frame = results[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )


@app.get("/video")
def video_feed():
    """
    Endpoint for streaming the raw webcam video.
    """
    return StreamingResponse(generate_raw_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/video/predictions")
def video_predictions():
    """
    Endpoint for streaming the webcam video with YOLOv8 predictions overlaid.
    """
    return StreamingResponse(generate_prediction_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")
