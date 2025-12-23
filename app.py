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
from transformers import CLIPProcessor, CLIPModel

# ================= CONFIG =================
CONF_THRESH = 0.3
CLIP_THRESHOLD = 0.6   # plastic confidence
MODEL_PATH = "yolov8n.pt"
# =========================================

# Thread pool
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# Load YOLO
yolo = YOLO(MODEL_PATH)

# Load CLIP (pretrained)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# Prompts
TEXT_PROMPTS = ["a plastic object", "a non plastic object"]

latest_result = {"plastic": 0, "confidence": 0.0, "timestamp": None}
raw_frame = None
current_future = None

app = FastAPI()


# ============ CLIP PLASTIC CLASSIFIER ============
def is_plastic_clip(crop):
    image = Image.fromarray(crop)

    inputs = clip_processor(
        text=TEXT_PROMPTS,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    plastic_prob = probs[0].item()
    return plastic_prob, plastic_prob > CLIP_THRESHOLD


# ============ YOLO + CLIP PIPELINE ============
def analyze_frame(jpg_bytes):
    global latest_result

    arr = np.frombuffer(jpg_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return

    img = cv2.resize(img, (640, 480))
    results = yolo(img, conf=CONF_THRESH, verbose=False)[0]

    plastic_found = 0
    confidence = 0.0

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        prob, is_plastic = is_plastic_clip(crop)

        if is_plastic:
            plastic_found = 1
            confidence = prob
            break

    latest_result = {
        "plastic": plastic_found,
        "confidence": round(confidence, 3),
        "timestamp": datetime.utcnow().isoformat()
    }

    print("Plastic:", plastic_found, "Confidence:", confidence)


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


@app.get("/health")
async def health():
    return {"status": "ok"}
