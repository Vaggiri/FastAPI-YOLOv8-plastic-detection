import asyncio
from datetime import datetime
import concurrent.futures

import cv2
import numpy as np
import torch
from PIL import Image

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO
from torchvision import models, transforms

# ================= CONFIG =================
CONF_THRESH = 0.3
MODEL_PATH = "yolov8n.pt"
MIN_AREA = 500
TEXTURE_THRESH = 0.25
# =========================================

executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# -------- YOLO --------
yolo = YOLO(MODEL_PATH)

# -------- MobileNet (pretrained, lightweight) --------
mobilenet = models.mobilenet_v3_small(pretrained=True)
mobilenet.eval()

feature_extractor = torch.nn.Sequential(
    mobilenet.features,
    mobilenet.avgpool
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ============== GLOBAL STATE ==============
latest_result = {
    "object": 0,
    "plastic": 0,
    "timestamp": None
}

trash_data = {
    "count": 0,
    "last_updated": None
}

raw_frame = None
current_future = None

app = FastAPI()

# ============ PLASTIC HEURISTIC ============
def is_plastic_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    img = Image.fromarray(crop)
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        feat = feature_extractor(x)
        feat_score = feat.abs().mean().item()

    return lap_var < 120 and feat_score > TEXTURE_THRESH


# ============ YOLO PIPELINE ============
def analyze_frame(jpg_bytes):
    global latest_result

    arr = np.frombuffer(jpg_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return

    img = cv2.resize(img, (640, 480))
    results = yolo(img, conf=CONF_THRESH, verbose=False)[0]

    object_detected = 1 if len(results.boxes) > 0 else 0
    plastic_found = 0

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)

        if area < MIN_AREA:
            continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        if is_plastic_crop(crop):
            plastic_found = 1
            break

    latest_result = {
        "object": object_detected,
        "plastic": plastic_found,
        "timestamp": datetime.utcnow().isoformat()
    }

    print("Object:", object_detected, "| Plastic:", plastic_found)


# ================= API =================
@app.post("/upload")
async def upload(request: Request):
    global raw_frame, current_future

    jpg = await request.body()
    if not jpg:
        return JSONResponse({"error": "empty frame"}, status_code=400)

    raw_frame = jpg

    if current_future and not current_future.done():
        return {"status": "skipped"}

    loop = asyncio.get_event_loop()
    current_future = loop.run_in_executor(executor, analyze_frame, jpg)

    return {"status": "ok"}


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
            await asyncio.sleep(0.01)

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/result")
async def result():
    return latest_result


# ============ TRASH COUNT =================
@app.post("/trashcount")
async def update_trashcount(data: dict):
    trash_data["count"] = data.get("count", trash_data["count"])
    trash_data["last_updated"] = datetime.utcnow().isoformat()
    return {"status": "ok", "trash": trash_data}


@app.get("/trashcount")
async def get_trashcount():
    return trash_data


@app.get("/health")
async def health():
    return {"status": "ok"}
