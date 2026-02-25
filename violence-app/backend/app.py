from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
import uuid

from model import predict_video

app = FastAPI()

UPLOAD_FOLDER = "temp_videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    unique_name = f"{uuid.uuid4()}.mp4"
    file_path = os.path.join(UPLOAD_FOLDER, unique_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    label, confidence, output_video = predict_video(file_path)

    return JSONResponse({
        "prediction": label,
        "confidence": round(confidence * 100, 2),
        "processed_video_path": output_video
    })