from fastapi import FastAPI, UploadFile, File, Request
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

app.mount("/videos", StaticFiles(directory=PROCESSED_FOLDER), name="videos")

@app.post("/predict/")
async def predict(request: Request, file: UploadFile = File(...)):

    unique_name = f"{uuid.uuid4()}.mp4"
    file_path = os.path.join(UPLOAD_FOLDER, unique_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    label, confidence, output_filename = predict_video(file_path)

    base_url = str(request.base_url)

    return JSONResponse({
        "prediction": label,
        "confidence": round(confidence * 100, 2),
        "video_url": f"{base_url}videos/{output_filename}"
    })

    