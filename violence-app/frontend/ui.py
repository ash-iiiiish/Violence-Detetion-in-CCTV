import streamlit as st
import requests
import os

BACKEND_URL = "http://127.0.0.1:8000/predict/"

st.set_page_config(page_title="AI Violence Detection", layout="wide")

st.title("🛡 AI-Powered Violence Detection System")

# ================= DEMO VIDEO =================
st.subheader("🎥 Try Demo Videos")

demo_folder = "C:/Users/kumar/OneDrive/Desktop/TRY-3/Violence-Detetion-in-CCTV/violence-app/demo-videos"
demo_files = [f for f in os.listdir(demo_folder) if f.endswith(".mp4")] if os.path.exists(demo_folder) else []

selected_demo = st.selectbox("Choose Demo Video", ["None"] + demo_files)

if selected_demo != "None":
    video_path = os.path.join(demo_folder, selected_demo)
    st.video(video_path)

    if st.button("Analyze Demo Video"):
        with st.spinner("Processing video... Please wait ⏳"):
            with open(video_path, "rb") as f:
                response = requests.post(
                    BACKEND_URL,
                    files={"file": (selected_demo, f, "video/mp4")}
                )

        result = response.json()

        # 🔥 ONLY SHOW PROCESSED VIDEO
        st.video(result["video_url"])

st.divider()

# ================= UPLOAD VIDEO =================
st.subheader("📤 Upload Your Own Video")
uploaded_file = st.file_uploader("Upload .mp4 Video", type=["mp4"])

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Analyze Uploaded Video"):
        with st.spinner("Processing video... Please wait ⏳"):
            response = requests.post(
                BACKEND_URL,
                files={"file": (uploaded_file.name, uploaded_file, "video/mp4")}
            )

        result = response.json()

        # 🔥 ONLY SHOW PROCESSED VIDEO
        st.video(result["video_url"])