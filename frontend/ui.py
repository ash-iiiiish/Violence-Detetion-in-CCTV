import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000/predict/"

st.set_page_config(page_title="AI Violence Detection", layout="wide")
st.title("🛡 AI-Powered Violence Detection System")

uploaded_file = st.file_uploader("Upload .mp4 Video", type=["mp4"])

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Analyze Video"):
        with st.spinner("Processing..."):
            response = requests.post(
                BACKEND_URL,
                files={"file": (uploaded_file.name, uploaded_file, "video/mp4")}
            )

        if response.status_code == 200:
            result = response.json()
            st.video(result["video_url"])
        else:
            st.error("Backend error occurred")