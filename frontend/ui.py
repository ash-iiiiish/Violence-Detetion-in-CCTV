import streamlit as st
import requests
import base64
import os
import subprocess
import tempfile

BACKEND_URL = "http://127.0.0.1:8000/predict/"

st.set_page_config(
    page_title="VIGIL.AI — Automated Violence Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Load demo image as base64 ─────────────────────────
def load_image_b64(paths):
    for path in paths:
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except Exception:
            continue
    return None

demo_img_b64 = load_image_b64([
    "images/demo.png",
    "images/demo.jpg",
    "images/1775591177571_image.png",
    "images/1775591177571_image.jpg",
])


# ── ffmpeg re-encode helper ───────────────────────────
def reencode_video_for_browser(video_bytes: bytes) -> bytes:
    """
    Re-encodes video bytes to H.264/AAC MP4 with faststart flag so the
    browser can play it inline without downloading the full file first.
    Returns the re-encoded bytes, or the original bytes if ffmpeg fails.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
            tmp_in.write(video_bytes)
            in_path = tmp_in.name

        out_path = in_path.replace(".mp4", "_browser.mp4")

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", in_path,
                "-vcodec", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-acodec", "aac",
                "-movflags", "+faststart",
                out_path,
            ],
            capture_output=True,
        )

        if result.returncode == 0 and os.path.exists(out_path):
            with open(out_path, "rb") as f:
                encoded = f.read()
        else:
            encoded = video_bytes          # fallback to original
    except Exception:
        encoded = video_bytes              # fallback if ffmpeg not found
    finally:
        try:
            os.unlink(in_path)
        except Exception:
            pass
        try:
            os.unlink(out_path)
        except Exception:
            pass

    return encoded


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Manrope:wght@700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #f0f2f5 !important;
    color: #0f172a !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 16px !important;
}

[data-testid="stAppViewContainer"] { background: #f0f2f5 !important; }
[data-testid="stHeader"]           { background: transparent !important; }
[data-testid="stToolbar"]          { display: none !important; }
#MainMenu, footer, header          { visibility: hidden !important; }
.stDeployButton                    { display: none !important; }

/* ── CONTAINER ── */
.block-container {
    padding: 0 3rem 5rem !important;
    max-width: 1500px !important;
}

@media (max-width: 1200px) { .block-container { padding: 0 2rem 4rem !important; } }
@media (max-width: 768px)  { .block-container { padding: 0 1rem 3rem !important; } }

::-webkit-scrollbar            { width: 4px; }
::-webkit-scrollbar-track      { background: #e2e8f0; }
::-webkit-scrollbar-thumb      { background: #cbd5e1; border-radius: 4px; }

/* ═══════════════════════════════
   NAV BAR
═══════════════════════════════ */
.vigil-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.3rem 0 1.3rem;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 2.5rem;
    flex-wrap: wrap;
    gap: 0.8rem;
    background: #f0f2f5;
    position: sticky;
    top: 0;
    z-index: 100;
}

.vigil-logo-area {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex-shrink: 0;
}

.vigil-eye-icon { width: 38px; height: 38px; flex-shrink: 0; }

.vigil-logo {
    font-family: 'Manrope', sans-serif;
    font-size: 1.65rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    color: #0f172a;
    line-height: 1;
}

.vigil-logo-dot { color: #2563eb; }

.vigil-nav-links {
    display: flex;
    align-items: center;
    gap: 2rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    font-weight: 500;
    color: #64748b;
}

.vigil-nav-links a {
    color: #64748b;
    text-decoration: none;
    transition: color 0.2s;
}
.vigil-nav-links a:hover { color: #0f172a; }
.vigil-nav-links .active { color: #2563eb; font-weight: 600; }

.nav-cta {
    display: flex;
    align-items: center;
    gap: 0.8rem;
}

.nav-status-pill {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    color: #16a34a;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 999px;
    padding: 0.32rem 0.85rem;
}

.nav-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 5px rgba(34,197,94,0.5);
    animation: navpulse 2.5s ease-in-out infinite;
}

@keyframes navpulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.35; }
}

/* ═══════════════════════════════
   HERO — TWO-COLUMN LAYOUT
═══════════════════════════════ */
.vigil-hero-wrap {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 2.5rem;
    margin-bottom: 2rem;
    padding: 2rem 0 1.5rem;
    flex-wrap: wrap;
}

.vigil-hero-text {
    flex: 1;
    min-width: 280px;
    max-width: 600px;
}

.vigil-hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    color: #2563eb;
    text-transform: uppercase;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 999px;
    padding: 0.32rem 0.9rem;
    margin-bottom: 1.1rem;
}

.vigil-hero-eyebrow::before {
    content: '●';
    font-size: 0.45rem;
    animation: navpulse 2s infinite;
}

.vigil-hero-title {
    font-family: 'Manrope', sans-serif;
    font-size: clamp(2rem, 4vw, 3.4rem);
    font-weight: 900;
    color: #0f172a;
    line-height: 1.12;
    letter-spacing: -0.03em;
    margin-bottom: 0.9rem;
}

.vigil-hero-title span { color: #2563eb; }

.vigil-hero-desc {
    font-size: clamp(0.9rem, 1.4vw, 1rem);
    color: #64748b;
    line-height: 1.75;
    font-weight: 400;
}

/* Demo image panel */
.vigil-hero-image {
    flex: 1;
    min-width: 280px;
    max-width: 620px;
    position: relative;
}

.vigil-hero-image-inner {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    position: relative;
}

.vigil-hero-image-inner img {
    width: 100%;
    height: auto;
    display: block;
    border-radius: 15px;
}

.vigil-hero-img-badge {
    position: absolute;
    top: 10px;
    left: 10px;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    font-weight: 600;
    color: #ffffff;
    background: rgba(15,23,42,0.75);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 6px;
    padding: 0.25rem 0.6rem;
    letter-spacing: 0.1em;
    backdrop-filter: blur(4px);
}

.vigil-hero-img-badge::before {
    content: '';
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #ef4444;
    box-shadow: 0 0 6px rgba(239,68,68,0.6);
    animation: navpulse 1.5s infinite;
}

.vigil-hero-img-placeholder {
    background: linear-gradient(135deg, #eff6ff 0%, #e0e7ff 100%);
    border-radius: 15px;
    min-height: 220px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.6rem;
    padding: 2rem;
    text-align: center;
}

.vigil-hero-img-placeholder .ph-icon { font-size: 2.5rem; opacity: 0.3; margin-bottom: 0.2rem; }

.vigil-hero-img-placeholder .ph-text {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #94a3b8;
    letter-spacing: 0.1em;
}

/* ═══════════════════════════════
   STATUS STRIP
═══════════════════════════════ */
.status-strip {
    display: flex;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    margin-bottom: 2rem;
    overflow: hidden;
    flex-wrap: wrap;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}

.strip-item {
    flex: 1;
    min-width: 130px;
    padding: 1rem 1.4rem;
    border-right: 1px solid #f1f5f9;
    border-bottom: 0;
    transition: background 0.2s;
}
.strip-item:hover { background: #f8fafc; }

@media (max-width: 900px) {
    .strip-item { min-width: 45%; border-bottom: 1px solid #f1f5f9; }
    .strip-item:nth-child(2n) { border-right: none; }
}
@media (max-width: 480px) {
    .strip-item { min-width: 100%; border-right: none; }
    .strip-item:last-child { border-bottom: none; }
}
.strip-item:last-child { border-right: none; }

.strip-key {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    color: #94a3b8;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}

.strip-val {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    color: #334155;
}

.strip-val.online { color: #16a34a; }

/* ═══════════════════════════════
   CARDS
═══════════════════════════════ */
.card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.5rem 1.7rem;
    position: relative;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}

@media (max-width: 600px) { .card { padding: 1.1rem; } }

.card-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #94a3b8;
    text-transform: uppercase;
    margin-bottom: 1.1rem;
    padding-bottom: 0.85rem;
    border-bottom: 1px solid #f1f5f9;
    display: flex;
    align-items: center;
    gap: 0.55rem;
}

.card-title::before {
    content: '';
    display: inline-block;
    width: 3px; height: 14px;
    background: #2563eb;
    border-radius: 2px;
    flex-shrink: 0;
}

/* ═══════════════════════════════
   FILE UPLOADER — FORCE LIGHT THEME
   FIX: hide the duplicate label text
═══════════════════════════════ */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploader"] > div > div,
[data-testid="stFileDropzone"],
section[data-testid="stFileUploader"] > div {
    background: #f8fafc !important;
    background-color: #f8fafc !important;
    color: #334155 !important;
}

/* Hide the top label that causes the duplicate "uploadUpload" text */
[data-testid="stFileUploader"] > label,
[data-testid="stFileUploader"] label:first-of-type {
    display: none !important;
    height: 0 !important;
    overflow: hidden !important;
}

[data-testid="stFileUploader"] > div {
    background: #f8fafc !important;
    border: 1.5px dashed #cbd5e1 !important;
    border-radius: 12px !important;
    transition: all 0.25s ease !important;
    padding: 1.8rem 1.2rem !important;
}

[data-testid="stFileUploader"] > div:hover {
    border-color: #2563eb !important;
    background: #eff6ff !important;
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] label span,
[data-testid="stFileUploader"] p,
[data-testid="stFileDropzone"] p,
[data-testid="stFileUploader"] section p,
[data-testid="stFileUploader"] div span {
    color: #64748b !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}

[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] .uploadedFile {
    color: #94a3b8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
}

[data-testid="stFileUploader"] * {
    background-color: transparent !important;
    color: #334155 !important;
}

[data-testid="stFileUploader"] > div {
    background-color: #f8fafc !important;
}

/* Browse files button */
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] [kind="secondary"],
[data-testid="stBaseButton-secondary"] {
    background: #ffffff !important;
    color: #2563eb !important;
    border: 1.5px solid #bfdbfe !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.1rem !important;
    box-shadow: none !important;
    transition: all 0.2s !important;
}

[data-testid="stFileUploader"] button:hover {
    background: #eff6ff !important;
    border-color: #2563eb !important;
}

/* ═══════════════════════════════
   BUTTON
═══════════════════════════════ */
.stButton > button {
    width: 100% !important;
    background: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.96rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    padding: 0.9rem 2rem !important;
    transition: all 0.2s ease !important;
    margin-top: 0.4rem !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.3) !important;
}

.stButton > button:hover {
    background: #1d4ed8 !important;
    box-shadow: 0 6px 22px rgba(37,99,235,0.4) !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active { transform: translateY(0) !important; }

/* ═══════════════════════════════
   VIDEO
═══════════════════════════════ */
[data-testid="stVideo"] video, video {
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    width: 100% !important;
    background: #f8fafc !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}

/* ═══════════════════════════════
   GLOBE LOADER
═══════════════════════════════ */
.globe-overlay {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 1.5rem;
    padding: 2.5rem 1rem 2rem;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

.globe-wrap {
    position: relative;
    width: 100px; height: 100px;
    flex-shrink: 0;
}

.globe-ring {
    position: absolute; inset: 0;
    border-radius: 50%;
    border: 1.5px solid rgba(37,99,235,0.15);
}

.globe-arc {
    position: absolute; inset: 0;
    border-radius: 50%;
    border: 2px solid transparent;
    border-top-color: #2563eb;
    border-right-color: rgba(37,99,235,0.3);
    animation: globeSpin 1.4s linear infinite;
    box-shadow: 0 0 16px rgba(37,99,235,0.15);
}

.globe-arc2 {
    position: absolute; inset: 12px;
    border-radius: 50%;
    border: 1.5px solid transparent;
    border-bottom-color: #60a5fa;
    border-left-color: rgba(96,165,250,0.3);
    animation: globeSpinR 2s linear infinite;
}

.globe-core {
    position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
}

.globe-core-dot {
    width: 16px; height: 16px;
    border-radius: 50%;
    background: radial-gradient(circle, #3b82f6, #2563eb 70%);
    box-shadow: 0 0 16px rgba(37,99,235,0.5), 0 0 32px rgba(37,99,235,0.15);
    animation: corePulse 1.4s ease-in-out infinite;
}

.globe-line {
    position: absolute; top: 50%; left: 50%;
    width: 40px; height: 1px;
    background: linear-gradient(90deg, rgba(37,99,235,0.5), transparent);
    transform-origin: left center;
    animation: radarSweep 2s linear infinite;
}
.globe-line:nth-child(5) { animation-delay: -0.66s; opacity: 0.5; }
.globe-line:nth-child(6) { animation-delay: -1.33s; opacity: 0.3; }

@keyframes globeSpin  { to { transform: rotate(360deg); } }
@keyframes globeSpinR { to { transform: rotate(-360deg); } }
@keyframes corePulse  {
    0%,100% { box-shadow: 0 0 16px rgba(37,99,235,0.5),0 0 32px rgba(37,99,235,0.15); }
    50%     { box-shadow: 0 0 26px rgba(37,99,235,0.8),0 0 50px rgba(37,99,235,0.25); }
}
@keyframes radarSweep { to { transform: rotate(360deg); } }

.globe-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    font-weight: 500;
    color: #2563eb;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    text-align: center;
}

.globe-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #94a3b8;
    letter-spacing: 0.08em;
    text-align: center;
    animation: textBlink 1.6s ease-in-out infinite;
    margin-top: 0.3rem;
}

@keyframes textBlink {
    0%,100% { opacity: 1; }
    50%     { opacity: 0.4; }
}

.globe-steps {
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
    width: 100%;
    max-width: 270px;
}

.globe-step {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #cbd5e1;
}

.globe-step-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #e2e8f0;
    flex-shrink: 0;
}

.globe-step.active .globe-step-dot {
    background: #2563eb;
    box-shadow: 0 0 6px rgba(37,99,235,0.4);
}
.globe-step.active { color: #475569; }
.globe-step.done .globe-step-dot { background: #22c55e; }
.globe-step.done  { color: #94a3b8; }

/* ── ANALYSIS COMPLETE BADGE ── */
.analysis-done {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.75rem 1.2rem;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 10px;
    margin-bottom: 0.5rem;
    animation: fadeUp 0.4s ease;
}

.analysis-done-icon {
    width: 24px; height: 24px;
    border-radius: 50%;
    background: #dcfce7;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem;
    color: #16a34a;
    flex-shrink: 0;
}

.analysis-done-text {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    color: #15803d;
    letter-spacing: 0.04em;
}

.analysis-done-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.63rem;
    color: #86efac;
    margin-top: 0.1rem;
}

/* ═══════════════════════════════
   RESULT CARD
═══════════════════════════════ */
.result-card {
    margin-top: 1.2rem;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    overflow: hidden;
    animation: fadeUp 0.4s ease;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    background: #ffffff;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

.result-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 1.5rem;
    background: #f8fafc;
    border-bottom: 1px solid #f1f5f9;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.result-top-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    color: #94a3b8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.result-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    padding: 0.3rem 0.9rem;
    border-radius: 6px;
    text-transform: uppercase;
}

.badge-safe   { color:#16a34a; border:1px solid #bbf7d0;  background:#f0fdf4; }
.badge-fight  { color:#b45309; border:1px solid #fcd34d; background:#fffbeb; }
.badge-weapon {
    color:#dc2626; border:1px solid #fca5a5; background:#fef2f2;
    animation: threatpulse 1.2s ease-in-out infinite;
}

@keyframes threatpulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(220,38,38,0); }
    50%     { box-shadow: 0 0 0 4px rgba(220,38,38,0.12); }
}

.result-stats {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    background: #ffffff;
}

@media (max-width: 600px) {
    .result-stats { grid-template-columns: 1fr 1fr; }
    .r-stat:last-child { grid-column: span 2; border-right: none; border-top: 1px solid #f1f5f9; }
}

.r-stat { padding: 1.2rem 1.5rem; border-right: 1px solid #f1f5f9; }
.r-stat:last-child { border-right: none; }

.r-stat-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.63rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #94a3b8;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

.r-stat-value {
    font-family: 'Manrope', sans-serif;
    font-size: 1.55rem;
    font-weight: 900;
    color: #0f172a;
    line-height: 1;
}

.r-stat-value.c-safe    { color: #16a34a; }
.r-stat-value.c-warning { color: #b45309; }
.r-stat-value.c-danger  { color: #dc2626; }

.r-stat-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #cbd5e1;
    margin-top: 0.3rem;
}

.conf-section {
    padding: 1rem 1.5rem 1.3rem;
    background: #f8fafc;
    border-top: 1px solid #f1f5f9;
}

.conf-header {
    display: flex;
    justify-content: space-between;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #94a3b8;
    margin-bottom: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.conf-header span:last-child { color: #334155; font-size: 0.8rem; font-weight: 600; }

.conf-track { height: 6px; background: #e2e8f0; border-radius: 999px; overflow: hidden; }
.conf-fill  { height: 100%; border-radius: 999px; }

/* ═══════════════════════════════
   IDLE BOXES
═══════════════════════════════ */
.idle-box {
    background: #f8fafc;
    border: 1.5px dashed #e2e8f0;
    border-radius: 12px;
    min-height: 160px;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 0.5rem; padding: 2rem; text-align: center;
}

.idle-icon  { font-size: 1.6rem; opacity: 0.25; margin-bottom: 0.2rem; }
.idle-title { font-family: 'DM Sans', sans-serif; font-size: 0.9rem; font-weight: 600; color: #94a3b8; }
.idle-sub   { font-family: 'DM Mono', monospace; font-size: 0.67rem; color: #cbd5e1; letter-spacing: 0.08em; }

/* ═══════════════════════════════
   DETECTION TABLE
═══════════════════════════════ */
.legend-table { width: 100%; border-collapse: collapse; margin-top: 0.3rem; }

.legend-table tr { border-bottom: 1px solid #f8fafc; transition: background 0.15s; }
.legend-table tr:last-child { border-bottom: none; }
.legend-table tr:hover { background: #f8fafc; }
.legend-table td { padding: 0.82rem 0.4rem; vertical-align: middle; }

.l-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 0.6rem;
    vertical-align: middle;
    flex-shrink: 0;
}

.l-class {
    font-family: 'DM Sans', sans-serif;
    font-weight: 700; color: #1e293b; font-size: 0.9rem;
}

.l-def {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem; font-weight: 400; color: #64748b;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 0.2rem 0.55rem;
    display: inline-block;
    margin-right: 0.4rem;
}

.l-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem; font-weight: 700;
    padding: 0.17rem 0.45rem;
    border-radius: 5px; letter-spacing: 0.08em;
    vertical-align: middle;
}

/* ═══════════════════════════════
   ERROR
═══════════════════════════════ */
.err-box {
    border: 1px solid #fca5a5;
    background: #fef2f2;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem; color: #dc2626; line-height: 1.7;
    margin-top: 1.2rem;
}

/* ═══════════════════════════════
   HELP & SUPPORT BAR
═══════════════════════════════ */
.help-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1rem 1.5rem;
    margin-top: 1.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    flex-wrap: wrap;
    gap: 1rem;
}

.help-bar-left {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.help-bar-icon {
    width: 36px; height: 36px;
    border-radius: 9px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}

.help-bar-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    color: #1e293b;
}

.help-bar-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.63rem;
    color: #94a3b8;
    letter-spacing: 0.07em;
    margin-top: 0.1rem;
}

.help-bar-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: #2563eb;
    color: #ffffff !important;
    text-decoration: none !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 0.6rem 1.3rem;
    border-radius: 9px;
    border: none;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 3px 12px rgba(37,99,235,0.25);
    white-space: nowrap;
    flex-shrink: 0;
}

.help-bar-btn:hover {
    background: #1d4ed8 !important;
    box-shadow: 0 5px 18px rgba(37,99,235,0.35);
    transform: translateY(-1px);
    color: #ffffff !important;
    text-decoration: none !important;
}

.help-bar-btn svg {
    width: 14px; height: 14px;
    flex-shrink: 0;
}

/* ═══════════════════════════════
   FOOTER
═══════════════════════════════ */
.vigil-divider { height: 1px; background: #e2e8f0; margin: 1.5rem 0 0; }

.vigil-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem; color: #cbd5e1; letter-spacing: 0.1em;
    padding: 0.9rem 0 0.5rem;
    flex-wrap: wrap; gap: 0.4rem;
}

/* ═══════════════════════════════
   CCTV BADGE STRIP
═══════════════════════════════ */
.cctv-badge-strip {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}

.cctv-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.63rem;
    font-weight: 500;
    color: #64748b;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 0.28rem 0.65rem;
    letter-spacing: 0.07em;
}

@media (max-width: 768px) {
    [data-testid="column"] { min-width: 100% !important; }
    .stHorizontalBlock { flex-wrap: wrap !important; }
    .vigil-hero-wrap { flex-direction: column; }
    .vigil-hero-image { max-width: 100%; }
}
</style>
""", unsafe_allow_html=True)


# ── NAV ──────────────────────────────────────────────
st.markdown("""
<div class="vigil-nav">
    <div class="vigil-logo-area">
        <svg class="vigil-eye-icon" viewBox="0 0 44 44" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="22" cy="22" r="20" stroke="rgba(37,99,235,0.12)" stroke-width="1"/>
            <circle cx="22" cy="22" r="14" stroke="rgba(37,99,235,0.18)" stroke-width="1"/>
            <circle cx="22" cy="22" r="20" stroke="#2563eb" stroke-width="1.5"
                    stroke-dasharray="28 97" stroke-linecap="round">
                <animateTransform attributeName="transform" type="rotate"
                    from="0 22 22" to="360 22 22" dur="3s" repeatCount="indefinite"/>
            </circle>
            <path d="M6 22 C10 13, 34 13, 38 22 C34 31, 10 31, 6 22Z"
                  fill="rgba(37,99,235,0.06)" stroke="#2563eb" stroke-width="1.2"/>
            <circle cx="22" cy="22" r="6" fill="rgba(37,99,235,0.1)" stroke="#2563eb" stroke-width="1.2"/>
            <circle cx="22" cy="22" r="2.5" fill="#2563eb">
                <animate attributeName="r" values="2.5;3.2;2.5" dur="2s" repeatCount="indefinite"/>
            </circle>
            <line x1="2" y1="8" x2="8" y2="2" stroke="rgba(37,99,235,0.3)" stroke-width="1.2" stroke-linecap="round"/>
            <line x1="42" y1="8" x2="36" y2="2" stroke="rgba(37,99,235,0.3)" stroke-width="1.2" stroke-linecap="round"/>
            <line x1="2" y1="36" x2="8" y2="42" stroke="rgba(37,99,235,0.3)" stroke-width="1.2" stroke-linecap="round"/>
            <line x1="42" y1="36" x2="36" y2="42" stroke="rgba(37,99,235,0.3)" stroke-width="1.2" stroke-linecap="round"/>
        </svg>
        <span class="vigil-logo">VIGIL<span class="vigil-logo-dot">.</span>AI</span>
    </div>
    <div class="vigil-nav-links">
        <a href="#" class="active">Dashboard</a>
        <a href="#">Models</a>
        <a href="#">Reports</a>
        <a href="#">Settings</a>
    </div>
    <div class="nav-cta">
        <div class="nav-status-pill"><span class="nav-dot"></span>System Online</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── HERO — two-column: text LEFT, image RIGHT ─────────
HERO_TEXT = """
<div class="vigil-hero-text">
    <div class="vigil-hero-eyebrow">Automated Violence &amp; Weapon Detection</div>
    <div class="vigil-hero-title">Intelligent <span>Threat Analysis</span><br>for Surveillance Footage</div>
    <div class="vigil-hero-desc">
        Upload a surveillance video to classify violent activity and detect weapons in real time.
        Powered by a 3D-CNN (R3D-18) for action recognition and YOLOv8 for weapon localisation,
        with frame-level annotation and confidence scoring.
    </div>
</div>
"""

if demo_img_b64:
    st.markdown(
        '<div class="vigil-hero-wrap">'
        + HERO_TEXT
        + '<div class="vigil-hero-image">'
        '<div class="vigil-hero-image-inner">'
        '<img src="data:image/png;base64,' + demo_img_b64 + '" alt="VIGIL.AI Demo — Annotated detection output"/>'
        '<div class="vigil-hero-img-badge">&#9679; LIVE DETECTION SAMPLE</div>'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="vigil-hero-wrap">'
        + HERO_TEXT
        + """
        <div class="vigil-hero-image">
            <div class="vigil-hero-image-inner">
                <div class="vigil-hero-img-placeholder">
                    <div class="ph-icon">&#127909;</div>
                    <div class="ph-text">PLACE demo.png IN images/ FOLDER</div>
                </div>
            </div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── CCTV BADGE STRIP ─────────────────────────────────
st.markdown("""
<div class="cctv-badge-strip">
    <span class="cctv-badge">&#128249; R3D-18 · 4-Class</span>
    <span class="cctv-badge">&#128269; YOLOv8 · Weapon Detector</span>
    <span class="cctv-badge">&#9889; FastAPI · 127.0.0.1:8000</span>
    <span class="cctv-badge">&#127902; 5-Frame Majority Vote</span>
    <span class="cctv-badge">&#128336; Real-Time Inference</span>
</div>
""", unsafe_allow_html=True)

# ── STATUS STRIP ─────────────────────────────────────
st.markdown("""
<div class="status-strip">
    <div class="strip-item">
        <div class="strip-key">Model Status</div>
        <div class="strip-val online">&#9679; Loaded &amp; Ready</div>
    </div>
    <div class="strip-item">
        <div class="strip-key">Violence Classifier</div>
        <div class="strip-val">R3D-18 · 4-Class</div>
    </div>
    <div class="strip-item">
        <div class="strip-key">Weapon Detector</div>
        <div class="strip-val">YOLOv8 · 4-Class</div>
    </div>
    <div class="strip-item">
        <div class="strip-key">Inference Backend</div>
        <div class="strip-val">FastAPI · 127.0.0.1:8000</div>
    </div>
    <div class="strip-item">
        <div class="strip-key">Smoothing Window</div>
        <div class="strip-val">5 Frames · Majority Vote</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── COLUMNS ──────────────────────────────────────────
col_left, col_right = st.columns([1.05, 1], gap="large")

with col_left:
    st.markdown('<div class="card" style="margin-bottom:1.4rem"><div class="card-title">Input — Surveillance Feed</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="Upload Surveillance Feed (.mp4)",
        type=["mp4"],
        label_visibility="collapsed"   # ← FIX: "collapsed" removes label space entirely
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        st.markdown('<div class="card" style="margin-bottom:1.4rem"><div class="card-title">Preview — Source Feed</div>', unsafe_allow_html=True)
        st.video(uploaded_file)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="card-title">Detection Classes</div>
        <table class="legend-table">
            <tr>
                <td style="width:36%"><span class="l-dot" style="background:#22c55e;box-shadow:0 0 5px rgba(34,197,94,0.4)"></span><span class="l-class">NonFight</span></td>
                <td>
                    <span class="l-def">No violent activity detected</span>
                    <span class="l-badge" style="color:#16a34a;border:1px solid #bbf7d0;background:#f0fdf4">SAFE</span>
                </td>
            </tr>
            <tr>
                <td><span class="l-dot" style="background:#f59e0b;box-shadow:0 0 5px rgba(245,158,11,0.4)"></span><span class="l-class">Fight</span></td>
                <td>
                    <span class="l-def">Physical altercation between subjects</span>
                    <span class="l-badge" style="color:#b45309;border:1px solid #fcd34d;background:#fffbeb">HIGH</span>
                </td>
            </tr>
            <tr>
                <td><span class="l-dot" style="background:#fb923c;box-shadow:0 0 5px rgba(251,146,60,0.4)"></span><span class="l-class">HockeyFight</span></td>
                <td>
                    <span class="l-def">Sport-context violent confrontation</span>
                    <span class="l-badge" style="color:#c2410c;border:1px solid #fed7aa;background:#fff7ed">HIGH</span>
                </td>
            </tr>
            <tr>
                <td><span class="l-dot" style="background:#a78bfa;box-shadow:0 0 5px rgba(167,139,250,0.4)"></span><span class="l-class">MovieFight</span></td>
                <td>
                    <span class="l-def">Scripted / cinematic fight sequence</span>
                    <span class="l-badge" style="color:#7c3aed;border:1px solid #ddd6fe;background:#f5f3ff">MED</span>
                </td>
            </tr>
            <tr>
                <td><span class="l-dot" style="background:#ef4444;box-shadow:0 0 5px rgba(239,68,68,0.4)"></span><span class="l-class">Weaponized</span></td>
                <td>
                    <span class="l-def">Knife · Handgun · Rifle · Launcher</span>
                    <span class="l-badge" style="color:#dc2626;border:1px solid #fca5a5;background:#fef2f2">CRITICAL</span>
                </td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)


with col_right:
    st.markdown('<div class="card"><div class="card-title">Analysis Control</div>', unsafe_allow_html=True)

    if uploaded_file is None:
        st.markdown("""
        <div class="idle-box">
            <div class="idle-icon">&#128225;</div>
            <div class="idle-title">No Feed Uploaded</div>
            <div class="idle-sub">UPLOAD AN .MP4 FILE TO BEGIN</div>
        </div>
        """, unsafe_allow_html=True)
        analyze = False
    else:
        analyze = st.button("Run Threat Analysis", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None and analyze:

        loader_slot = st.empty()
        loader_slot.markdown("""
        <div class="globe-overlay">
            <div class="globe-wrap">
                <div class="globe-ring"></div>
                <div class="globe-arc"></div>
                <div class="globe-arc2"></div>
                <div class="globe-core"><div class="globe-core-dot"></div></div>
                <div class="globe-line"></div>
                <div class="globe-line"></div>
                <div class="globe-line"></div>
            </div>
            <div>
                <div class="globe-label">Analysing Threat Feed</div>
                <div class="globe-sub">VIGIL.AI — INFERENCE PIPELINE ACTIVE</div>
            </div>
            <div class="globe-steps">
                <div class="globe-step done">
                    <div class="globe-step-dot"></div>Video decoded &amp; cleaned
                </div>
                <div class="globe-step active">
                    <div class="globe-step-dot"></div>R3D-18 violence classification
                </div>
                <div class="globe-step active">
                    <div class="globe-step-dot"></div>YOLOv8 weapon detection
                </div>
                <div class="globe-step">
                    <div class="globe-step-dot"></div>Frame annotation &amp; export
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        try:
            uploaded_file.seek(0)
            response = requests.post(
                BACKEND_URL,
                files={"file": (uploaded_file.name, uploaded_file, "video/mp4")}
            )
            error_msg = None
        except Exception as e:
            response = None
            error_msg = str(e)

        loader_slot.markdown("""
        <div class="analysis-done">
            <div class="analysis-done-icon">&#10003;</div>
            <div>
                <div class="analysis-done-text">Analysis Complete</div>
                <div class="analysis-done-sub">All pipeline stages finished successfully</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if response is not None and response.status_code == 200:
            result     = response.json()
            prediction = result["prediction"]
            confidence = result["confidence"]
            video_url  = result["video_url"]
            pred_lower = prediction.lower()
            conf_int   = int(confidence)

            if "weapon" in pred_lower:
                badge_cls  = "badge-weapon"
                val_cls    = "c-danger"
                bar_color  = "#dc2626"
                threat_lv  = "CRITICAL"
                threat_col = "#dc2626"
            elif "fight" in pred_lower:
                badge_cls  = "badge-fight"
                val_cls    = "c-warning"
                bar_color  = "#f59e0b"
                threat_lv  = "HIGH"
                threat_col = "#b45309"
            else:
                badge_cls  = "badge-safe"
                val_cls    = "c-safe"
                bar_color  = "#22c55e"
                threat_lv  = "NONE"
                threat_col = "#16a34a"

            st.markdown(
                '<div class="result-card">'
                '<div class="result-top">'
                '<div class="result-top-label">Assessment Result</div>'
                '<div class="result-badge ' + badge_cls + '">' + prediction + '</div>'
                '</div>'
                '<div class="result-stats">'
                '<div class="r-stat">'
                '<div class="r-stat-label">Classification</div>'
                '<div class="r-stat-value ' + val_cls + '" style="font-size:1.05rem;margin-top:0.15rem">' + prediction + '</div>'
                '<div class="r-stat-sub">R3D-18 Output</div>'
                '</div>'
                '<div class="r-stat">'
                '<div class="r-stat-label">Confidence</div>'
                '<div class="r-stat-value ' + val_cls + '">' + str(confidence) + '%</div>'
                '<div class="r-stat-sub">Softmax Score</div>'
                '</div>'
                '<div class="r-stat">'
                '<div class="r-stat-label">Threat Level</div>'
                '<div class="r-stat-value" style="color:' + threat_col + '">' + threat_lv + '</div>'
                '<div class="r-stat-sub">Risk Assessment</div>'
                '</div>'
                '</div>'
                '<div class="conf-section">'
                '<div class="conf-header">'
                '<span>Confidence Score</span>'
                '<span>' + str(confidence) + '%</span>'
                '</div>'
                '<div class="conf-track">'
                '<div class="conf-fill" style="width:' + str(conf_int) + '%;background:linear-gradient(90deg,' + bar_color + '99,' + bar_color + ')"></div>'
                '</div>'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            # ── FIX: Fetch annotated video → re-encode via ffmpeg → display ──
            st.markdown(
                '<div class="card" style="margin-top:1.4rem">'
                '<div class="card-title">Output — Annotated Feed</div>',
                unsafe_allow_html=True,
            )

            try:
                vid_response = requests.get(video_url, timeout=60)
                if vid_response.status_code == 200:
                    raw_bytes     = vid_response.content
                    browser_bytes = reencode_video_for_browser(raw_bytes)
                    st.video(browser_bytes)
                else:
                    st.markdown(
                        '<div class="err-box">&#10005; &nbsp;Could not fetch annotated video — '
                        'HTTP ' + str(vid_response.status_code) + '</div>',
                        unsafe_allow_html=True,
                    )
            except Exception as fetch_err:
                st.markdown(
                    '<div class="err-box">&#10005; &nbsp;Video fetch error — ' + str(fetch_err) + '</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

        else:
            err = error_msg if response is None else f"HTTP {response.status_code}"
            st.markdown(
                '<div class="err-box">'
                '&#10005; &nbsp;Backend connection failed — ' + err + '<br>'
                'Ensure the FastAPI server is running at 127.0.0.1:8000'
                '</div>',
                unsafe_allow_html=True,
            )

    elif uploaded_file is not None:
        st.markdown("""
        <div class="card" style="margin-top:1.4rem">
            <div class="card-title">Output — Annotated Feed</div>
            <div class="idle-box">
                <div class="idle-icon">&#9654;</div>
                <div class="idle-title">Ready to Analyse</div>
                <div class="idle-sub">PRESS RUN THREAT ANALYSIS</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── HELP & SUPPORT BAR ────────────────────────────────
st.markdown("""
<div class="help-bar">
    <div class="help-bar-left">
        <div class="help-bar-icon">&#128663;</div>
        <div>
            <div class="help-bar-title">Help &amp; Support</div>
            <div class="help-bar-sub">QUESTIONS &nbsp;&middot;&nbsp; BUG REPORTS &nbsp;&middot;&nbsp; FEATURE REQUESTS</div>
        </div>
    </div>
    <a class="help-bar-btn"
       href="mailto:kumarashisha54@gmail.com?subject=VIGIL.AI%20Support%20Request&body=Hi%2C%0A%0AI%20need%20help%20with%20VIGIL.AI%3A%0A%0A"
       target="_blank">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round"
                  d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/>
        </svg>
        Contact Support
    </a>
</div>
""", unsafe_allow_html=True)

# ── FOOTER ───────────────────────────────────────────
st.markdown("""
<div class="vigil-divider"></div>
<div class="vigil-footer">
    <div>VIGIL.AI &nbsp;&middot;&nbsp; Automated Violence &amp; Weapon Detection &nbsp;&middot;&nbsp; R3D-18 + YOLOv8</div>
    <div>FIGHT &nbsp;&middot;&nbsp; HOCKEY FIGHT &nbsp;&middot;&nbsp; MOVIE FIGHT &nbsp;&middot;&nbsp; NON-FIGHT &nbsp;&middot;&nbsp; WEAPONIZED</div>
</div>
""", unsafe_allow_html=True)