from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import uuid

from model import predict_video

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "temp_videos")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed_videos")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Serve processed videos
app.mount("/videos", StaticFiles(directory=PROCESSED_FOLDER), name="videos")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    unique_name = f"{uuid.uuid4()}.mp4"
    file_path = os.path.join(UPLOAD_FOLDER, unique_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    label, confidence, output_filename = predict_video(file_path)

    return JSONResponse({
        "prediction": label,
        "confidence": round(confidence * 100, 2),
        "video_url": f"http://127.0.0.1:8000/videos/{output_filename}"
    })