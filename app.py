import io
import asyncio
from datetime import datetime
import concurrent.futures

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from ultralytics import YOLO
import cv2

# ================== CONFIG ==================
MODEL_PATH = "yolov8n.pt"  # use nano for Render (FAST)
CONF_THRESH = 0.35

# Plastic-related COCO classes
PLASTIC_CLASSES = {
    "bottle",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl"
}

# Minimum area to consider (filters noise)
MIN_BOX_AREA = 800  # tweak if needed

# ============================================

executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.model.names

latest_result = {"plastic": 0, "timestamp": None}
raw_frame = None
current_analysis_future = None

app = FastAPI()


# ================== YOLO ANALYSIS ==================
def sync_analyze_frame(jpeg_bytes):
    global latest_result

    arr = np.frombuffer(jpeg_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return

    # Resize → faster inference
    img = cv2.resize(img, (640, 480))

    results = model(img, conf=CONF_THRESH, imgsz=640, verbose=False)[0]

    plastic_found = 0

    for box in results.boxes:
        cls_id = int(box.cls)
        cls_name = CLASS_NAMES[cls_id]

        if cls_name not in PLASTIC_CLASSES:
            continue

        x1, y1, x2, y2 = box.xyxy[0]
        area = (x2 - x1) * (y2 - y1)

        if area >= MIN_BOX_AREA:
            plastic_found = 1
            break

    latest_result = {
        "plastic": plastic_found,
        "timestamp": datetime.utcnow().isoformat()
    }

    print("Plastic detected:", plastic_found)


# ================== UPLOAD ==================
@app.post("/upload")
async def upload(request: Request):
    global raw_frame, current_analysis_future

    jpg = await request.body()
    if not jpg:
        return JSONResponse({"error": "empty frame"}, status_code=400)

    # Always update latest frame (stream priority)
    raw_frame = jpg

    # Skip inference if model busy
    if current_analysis_future and not current_analysis_future.done():
        return {"status": "skipped"}

    loop = asyncio.get_event_loop()
    current_analysis_future = loop.run_in_executor(
        executor, sync_analyze_frame, jpg
    )

    return {"status": "ok"}


# ================== LOW LATENCY STREAM ==================
@app.get("/stream")
async def stream():
    async def gen():
        last = None
        while True:
            if raw_frame and raw_frame != last:
                last = raw_frame
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    raw_frame +
                    b"\r\n"
                )
            await asyncio.sleep(0.01)  # 🔥 keeps latency LOW

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ================== RESULT ==================
@app.get("/result")
async def result():
    return latest_result


@app.get("/health")
async def health():
    return {"status": "ok"}
