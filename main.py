import asyncio

import cv2
import numpy as np
from fastapi import FastAPI
from starlette.responses import StreamingResponse
from fastapi.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect
from ultralytics import YOLO
import queue
import zlib

app = FastAPI()

model = YOLO("model.pt")

streams = {}

def generate_raw_frames():
    cap = cv2.VideoCapture(0)
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

def get_stream_frames(stream_id):
    frame = streams[stream_id]

    return frame

def generate_prediction_frames(id: str):
    while True:
        if not id in streams:
            return b''

        frame_bytes = get_stream_frames(id)
        # convert bytes to cv2 image
        frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        results = model(source=frame)
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
    return StreamingResponse(generate_raw_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/video/predictions/{id}")
def video_predictions(id: str):
    return StreamingResponse(generate_prediction_frames(id),
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/stream/{id}")
async def stream(socket: WebSocket, id: str):
    print(f"Client connected to stream {id}")
    if not id in streams:
        streams[id] = b''

    await socket.accept()
    try:
        while True:
            frame_bytes = await socket.receive_bytes()
            # compressed = zlib.compress(frame_bytes, level=1)
            # streams[id].put(compressed)
            streams[id] = frame_bytes

    except WebSocketDisconnect:
        del streams[id]
