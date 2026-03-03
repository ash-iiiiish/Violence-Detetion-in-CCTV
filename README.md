# Violence-Detetion-in-CCTV
An AI-Powered System which can detect any type of violence occured in a video file that can be used in CCTV recordings to detect noise and provide necessary alerts.

## Introduction

Violence Detection in CCTV is an AI-powered surveillance system that
detects violent activities in video footage using Deep Learning and
Computer Vision.\
The system combines:

-   **R3D-18 (3D ResNet)** for temporal violence classification
-   **YOLOv8** for real-time human/object detection
-   **FastAPI** backend for video processing API
-   **Streamlit** frontend for user interaction

This project is designed for real-time CCTV monitoring, automated
alerts, and video-based violence detection.

------------------------------------------------------------------------

## Table of Contents

-   Introduction
-   Features
-   Project Architecture
-   Installation
-   Usage
-   API Endpoints
-   Configuration
-   Dependencies
-   Project Structure
-   Examples
-   Troubleshooting
-   License

------------------------------------------------------------------------

## Features

-   Real-time violence detection
-   Video upload and processing via API
-   YOLO-based person detection
-   3D CNN (R3D-18) temporal classification
-   Smoothed prediction window
-   Processed video output with detection results
-   Streamlit-based UI
-   GPU acceleration support (CUDA)

------------------------------------------------------------------------

## Project Architecture

1.  Video is uploaded through the frontend.
2.  FastAPI backend receives and stores the video.
3.  Frames are processed using:
    -   YOLOv8 for object detection
    -   R3D-18 for violence classification
4.  Predictions are smoothed using a sliding window.
5.  Output video is generated with annotations.
6.  Processed video is returned to the user.

------------------------------------------------------------------------

## Installation

### 1. Clone the Repository

``` bash
git clone https://github.com/your-username/Violence-Detection-in-CCTV.git
cd Violence-Detection-in-CCTV/violence-app
```

### 2. Create Virtual Environment

``` bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Usage

### Run Backend (FastAPI)

``` bash
cd backend
uvicorn app:app --reload
```

Backend runs at:

    http://127.0.0.1:8000

### Run Frontend (Streamlit)

``` bash
cd frontend
streamlit run ui.py
```

------------------------------------------------------------------------

## API Endpoints

### POST `/predict/`

Upload a video file for violence detection.

**Request:** - Form-data with video file

**Response:** - JSON containing processed video path

------------------------------------------------------------------------

## Configuration

You may need to update model paths inside:

-   `backend/model.py`
-   `live_model.py`

Update:

-   MODEL_PATH
-   YOLO_PATH
-   Input/output video paths

For GPU usage:

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

------------------------------------------------------------------------

## Dependencies

Main libraries used:

-   torch \>= 2.0.0
-   torchvision \>= 0.15.0
-   numpy
-   opencv-python
-   scikit-learn
-   ultralytics (YOLOv8)
-   fastapi
-   uvicorn
-   streamlit

Full list available in `requirements.txt`.

------------------------------------------------------------------------

## Project Structure

    Violence-Detetion-in-CCTV-main/
    │
    ├── violence-app/
    │   ├── backend/
    │   │   ├── app.py
    │   │   └── model.py
    │   ├── frontend/
    │   │   └── ui.py
    │   ├── live_model.py
    │   └── requirements.txt
    │
    ├── LICENSE
    └── README.md

------------------------------------------------------------------------

## Examples

1.  Upload a CCTV video through Streamlit UI.
2.  System detects violent activity.
3.  Output video is generated with bounding boxes and labels.
4.  Processed video available via `/videos/` endpoint.

------------------------------------------------------------------------

## Troubleshooting

### Model Not Loading

-   Ensure correct MODEL_PATH.
-   Verify model file exists.
-   Check CUDA availability.

### Video Not Processing

-   Confirm OpenCV installation.
-   Ensure video format is supported (MP4 recommended).

### YOLO Errors

-   Ensure ultralytics is installed properly.
-   Check YOLO model path.

------------------------------------------------------------------------

## 👨‍💻 Contributors
- [@ash-iiiiish](https://github.com/ash-iiiiish)


## 🤝 Contributing
Contributions are welcome! Fork this repository and submit a pull request.

small structural changes
small structural changes
small structural changes

