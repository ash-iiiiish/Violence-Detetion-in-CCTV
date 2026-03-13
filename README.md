# 🚨 AI-Powered Violence Detection in CCTV:

An AI-powered surveillance system that detects violent activity in CCTV
footage using Deep Learning and Computer Vision.

This project combines **spatio-temporal video understanding**, **object
detection**, and **pose-aware analysis** to identify potential violent
incidents in recorded or live surveillance videos.

------------------------------------------------------------------------

# 📌 Project Overview

Violence Detection in CCTV is designed to automatically analyze video
footage and detect violent activity using a combination of:

-   Fine-tuned **R3D-18 (3D ResNet)** for temporal violence
    classification
-   Fine-tuned **YOLOv8** for real-time person and weapon detection
-   **Pose estimation** for human posture analysis
-   **FFmpeg-based video processing pipeline**
-   **FastAPI backend** for model inference
-   **Streamlit frontend** for interactive user interface

The system processes uploaded CCTV videos, analyzes motion and object
interactions, and generates an annotated output video highlighting
detected violence.

------------------------------------------------------------------------

# ✨ Key Features

-   🎥 Violence detection from CCTV footage
-   🧠 Spatio-temporal video understanding using 3D CNN
-   🔍 Fine-tuned YOLOv8 for custom weapon detection (guns, rifles,
    etc.)
-   🧍 Pose-aware human motion analysis
-   🎬 Automated video processing pipeline using FFmpeg
-   ⚡ FastAPI-based backend for inference
-   🖥 Interactive Streamlit UI
-   🚀 GPU acceleration support (CUDA)

------------------------------------------------------------------------

# 🏗 System Architecture

The system follows a multi-stage video processing pipeline:

CCTV Video Input\
↓\
Video Preprocessing (FFmpeg)\
↓\
Frame Extraction\
↓\
YOLOv8 Detection (Persons / Weapons)\
↓\
Pose Estimation (Human Keypoints)\
↓\
R3D-18 Temporal Violence Classification\
↓\
Sliding Window Prediction Smoothing\
↓\
Annotated Output Video

------------------------------------------------------------------------

# ⚙️ Technologies Used

## AI / Deep Learning

-   PyTorch
-   R3D-18 (3D CNN)
-   YOLOv8 (Fine-tuned)
-   Pose Estimation

## Backend

-   FastAPI
-   Uvicorn

## Frontend

-   Streamlit

## Video Processing

-   OpenCV
-   FFmpeg

## Other Tools

-   NumPy
-   Scikit-learn

------------------------------------------------------------------------

# 📂 Project Structure

Violence-Detetion-in-CCTV\
│\
├── violence-app/\
│ ├── backend/\
│ │ ├── app.py\
│ │ └── model.py\
│ │\
│ ├── frontend/\
│ │ └── ui.py\
│ │\
│ ├── live_model.py\
│ └── requirements.txt\
│\
├── README.md\
└── LICENSE

------------------------------------------------------------------------

# 🚀 Installation

## 1 Clone Repository

git clone https://github.com/ash-iiiiish/Violence-Detetion-in-CCTV\
cd Violence-Detetion-in-CCTV/violence-app

------------------------------------------------------------------------

## 2 Create Virtual Environment

python -m venv venv

Activate:

Windows

venv`\Scripts`{=tex}`\activate`{=tex}

Linux / Mac

source venv/bin/activate

------------------------------------------------------------------------

## 3 Install Dependencies

pip install -r requirements.txt

------------------------------------------------------------------------

# ▶️ Running the Application

## Start Backend

cd backend\
uvicorn app:app --reload

Backend runs at:

http://127.0.0.1:8000

------------------------------------------------------------------------

## Start Frontend

cd frontend\
streamlit run ui.py

------------------------------------------------------------------------

# 📡 API Endpoint

### POST `/predict/`

Upload a video file for violence detection.

Input\
- Video file (form-data)

Output\
- JSON response containing processed video path

------------------------------------------------------------------------

# 📊 Example Workflow

1.  Upload CCTV footage using the Streamlit interface.\
2.  System processes the video through the AI pipeline.\
3.  Violence is detected using temporal motion analysis.\
4.  Annotated output video is generated with detection results.

------------------------------------------------------------------------

# ⚠️ Troubleshooting

### Model Not Loading

-   Verify correct model path in `model.py`
-   Ensure weights are available
-   Check CUDA availability

### Video Processing Errors

-   Ensure FFmpeg is installed
-   Confirm video format is supported (MP4 recommended)

------------------------------------------------------------------------


## 👨‍💻 Contributors
- [@ash-iiiiish](https://github.com/ash-iiiiish)
- [@rhitansh](https://github.com/rhitansh)

------------------------------------------------------------------------

## 🤝 Contributing
Contributions are welcome! Fork this repository and submit a pull request.


# ⭐ If you found this project interesting, consider giving it a star!


architectural changes 
architectural changes 
