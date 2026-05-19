import streamlit as st
import requests
import base64
import os
import subprocess
import tempfile
import textwrap
from datetime import datetime
import io


# ── HTML helper: dedent + strip so Streamlit's markdown parser
#    doesn't mistake indented HTML for a code block.
def html(s):
    return '\n'.join(line.strip() for line in textwrap.dedent(s).strip().splitlines())


def md(s):
    """Shorthand: st.markdown with unsafe HTML + dedent."""
    st.markdown(html(s), unsafe_allow_html=True)

BACKEND_URL = "http://127.0.0.1:8000/predict/"

st.set_page_config(
    page_title="VIGIL.AI — Automated Violence Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Session state ─────────────────────────────────────
if "analysis_result"    not in st.session_state: st.session_state.analysis_result    = None
if "analysis_video"     not in st.session_state: st.session_state.analysis_video     = None
if "analysis_filename"  not in st.session_state: st.session_state.analysis_filename  = "unknown.mp4"
if "analysis_timestamp" not in st.session_state: st.session_state.analysis_timestamp = None

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
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
            tmp_in.write(video_bytes)
            in_path = tmp_in.name
        out_path = in_path.replace(".mp4", "_browser.mp4")
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", in_path,
             "-vcodec", "libx264", "-preset", "fast", "-crf", "23",
             "-acodec", "aac", "-movflags", "+faststart", out_path],
            capture_output=True,
        )
        if result.returncode == 0 and os.path.exists(out_path):
            with open(out_path, "rb") as f:
                encoded = f.read()
        else:
            encoded = video_bytes
    except Exception:
        encoded = video_bytes
    finally:
        try: os.unlink(in_path)
        except: pass
        try: os.unlink(out_path)
        except: pass
    return encoded

# ── PDF Report Generator ──────────────────────────────
def generate_pdf_report(filename, prediction, confidence, threat_level, timestamp):
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_margins(20, 20, 20)
        pdf.set_auto_page_break(auto=True, margin=25)

        pdf.set_fill_color(37, 99, 235)
        pdf.rect(0, 0, 210, 32, 'F')
        pdf.set_font("Helvetica", "B", 17)
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(20, 7)
        pdf.cell(0, 12, "VIGIL.AI  Threat Analysis Report", ln=True)
        pdf.set_xy(20, 19)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(191, 219, 254)
        pdf.cell(0, 8, f"Generated: {timestamp}   |   Automated Violence & Weapon Detection System", ln=True)

        pdf.ln(12)

        def section_header(label):
            pdf.set_fill_color(239, 246, 255)
            pdf.set_draw_color(191, 219, 254)
            pdf.set_text_color(37, 99, 235)
            pdf.set_font("Helvetica", "B", 10)
            y = pdf.get_y()
            pdf.rect(20, y, 170, 9, 'FD')
            pdf.set_xy(24, y)
            pdf.cell(0, 9, label, ln=True)
            pdf.ln(2)

        def row(key, val):
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(100, 116, 139)
            pdf.cell(52, 8, key.upper(), border=0)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(0, 8, str(val), border=0, ln=True)
            pdf.set_draw_color(241, 245, 249)
            pdf.line(20, pdf.get_y(), 190, pdf.get_y())

        section_header("ANALYSIS SUMMARY")
        row("File Analysed", filename)
        row("Classification", prediction)
        row("Confidence Score", f"{confidence}%")
        row("Threat Level", threat_level)
        row("Analysis Timestamp", timestamp)
        row("Inference Pipeline", "R3D-18 Violence Classifier + YOLOv8 Weapon Detector")
        row("Temporal Smoothing", "5-Frame Majority Vote")

        pdf.ln(8)

        if threat_level == "CRITICAL":
            fr, fg, fb = 254, 242, 242
            tr2, tg2, tb2 = 220, 38, 38
            msg = "CRITICAL: Weapon detected in surveillance feed. Immediate intervention recommended."
        elif threat_level == "HIGH":
            fr, fg, fb = 255, 251, 235
            tr2, tg2, tb2 = 180, 83, 9
            msg = "HIGH: Physical altercation detected. Escalation monitoring advised."
        elif threat_level == "MODERATE":
            fr, fg, fb = 245, 243, 255
            tr2, tg2, tb2 = 109, 40, 217
            msg = "MODERATE: Scripted/cinematic fight sequence detected. Low real-world threat probability."
        else:
            fr, fg, fb = 240, 253, 244
            tr2, tg2, tb2 = 21, 128, 61
            msg = "NONE: No violent activity detected in this surveillance feed."

        y0 = pdf.get_y()
        pdf.set_fill_color(fr, fg, fb)
        pdf.set_draw_color(tr2, tg2, tb2)
        pdf.rect(20, y0, 170, 28, 'FD')
        pdf.set_xy(26, y0 + 4)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(tr2, tg2, tb2)
        pdf.cell(0, 8, f"Threat Level: {threat_level}", ln=True)
        pdf.set_xy(26, y0 + 13)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(51, 65, 85)
        pdf.multi_cell(160, 6, msg)

        pdf.ln(10)

        section_header("MODELS USED")
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(37, 99, 235)
        pdf.cell(0, 8, "R3D-18 — 3D ResNet Violence Classifier", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(71, 85, 105)
        pdf.multi_cell(0, 6,
            "Fine-tuned on a 4-class violence dataset (Fight, HockeyFight, MovieFight, NonFight). "
            "Uses 3D convolutions to capture spatiotemporal patterns across 16-frame clips. "
            "Outputs per-clip softmax scores aggregated via 5-frame majority vote for temporal stability.")

        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(37, 99, 235)
        pdf.cell(0, 8, "YOLOv8 — Weapon Detection Model", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(71, 85, 105)
        pdf.multi_cell(0, 6,
            "Fine-tuned YOLOv8 for 4-class weapon detection: Knife, Handgun, Rifle, Launcher. "
            "Applied frame-by-frame for precise bounding-box localisation. "
            "Triggers 'Weaponized' override when any weapon is detected above the confidence threshold.")

        pdf.ln(8)

        section_header("DISCLAIMER")
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(100, 116, 139)
        pdf.multi_cell(0, 5,
            "This report is generated automatically by the VIGIL.AI inference pipeline. "
            "Results are probabilistic and should be reviewed by a qualified security professional "
            "before any enforcement action is taken. VIGIL.AI is a decision-support tool, not a "
            "replacement for human judgment.")

        pdf.set_y(-22)
        pdf.set_draw_color(226, 232, 240)
        pdf.line(20, pdf.get_y(), 190, pdf.get_y())
        pdf.ln(3)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(148, 163, 184)
        pdf.cell(0, 5,
            "VIGIL.AI  ·  Automated Violence & Weapon Detection  ·  R3D-18 + YOLOv8  ·  Confidential",
            align="C")

        return bytes(pdf.output())

    except ImportError:
        text = (
            f"VIGIL.AI THREAT ANALYSIS REPORT\n"
            f"{'='*44}\n"
            f"Generated  : {timestamp}\n"
            f"File       : {filename}\n"
            f"Result     : {prediction}\n"
            f"Confidence : {confidence}%\n"
            f"Threat     : {threat_level}\n"
            f"Pipeline   : R3D-18 + YOLOv8\n"
            f"{'='*44}\n"
            f"Note: Install fpdf2 (pip install fpdf2) for full PDF reports.\n"
        )
        return text.encode("utf-8")


# ═══════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Manrope:wght@700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* ═══ SMOOTH ANCHOR SCROLLING ═══ */
html { scroll-behavior: smooth; }
.vigil-section {
    scroll-margin-top: 100px;   /* clears the sticky nav when scrolled to */
    display: block;
}
.vigil-section-spacer { height: 2.2rem; }

html, body, [data-testid="stAppViewContainer"] {
    background: #f0f2f5 !important;
    color: #0f172a !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 16px !important;
}

.model-para, .model-para *, .model-ft-text, .model-ft-text *,
.model-arch-val, .model-arch-key, .model-card-body p,
.model-card-body div, .model-card-body span,
.report-meta-val, .faq-a, .support-channel-title,
.card p, .card div, .card span {
    font-family: 'DM Sans', sans-serif !important;
}
.model-arch-key, .strip-key, .r-stat-label, .r-stat-sub,
.conf-header, .globe-label, .globe-sub, .globe-step,
.nav-status-pill, .vigil-hero-eyebrow, .card-title,
.cctv-badge, .vigil-footer, .page-header-eyebrow,
.model-section-title, .model-card-tag,
.model-ft-num, .report-meta-key, .support-channel-sub {
    font-family: 'DM Mono', monospace !important;
}
.vigil-logo, .vigil-hero-title, .model-card-name,
.model-hero-title, .settings-section-title, .page-header-title,
.r-stat-value {
    font-family: 'Manrope', sans-serif !important;
}
strong, b { font-family: inherit !important; font-weight: 700 !important; }

[data-testid="stAppViewContainer"] { background: #f0f2f5 !important; }
[data-testid="stHeader"]           { background: transparent !important; }
[data-testid="stToolbar"]          { display: none !important; }
#MainMenu, footer, header          { visibility: hidden !important; }
.stDeployButton                    { display: none !important; }

.block-container {
    padding: 0 3rem 5rem !important;
    max-width: 1500px !important;
}
@media (max-width: 1200px) { .block-container { padding: 0 2rem 4rem !important; } }
@media (max-width: 768px)  { .block-container { padding: 0 1rem 3rem !important; } }

::-webkit-scrollbar            { width: 4px; }
::-webkit-scrollbar-track      { background: #e2e8f0; }
::-webkit-scrollbar-thumb      { background: #cbd5e1; border-radius: 4px; }

/* ═══ HIDDEN NAV BUTTONS ═══ */
/* These are real Streamlit buttons that drive navigation */
div[data-testid="stHorizontalBlock"].nav-btn-row > div[data-testid="column"] > div > div > div > button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #64748b !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    padding: 0.25rem 0.1rem !important;
    width: auto !important;
    margin: 0 !important;
    min-width: unset !important;
    letter-spacing: 0 !important;
    transform: none !important;
}
div[data-testid="stHorizontalBlock"].nav-btn-row > div[data-testid="column"] > div > div > div > button:hover {
    color: #0f172a !important;
    background: transparent !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ═══ NAV ═══ */
.vigil-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.3rem 0 1.7rem;
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
    display: flex; align-items: center; gap: 0.75rem; flex-shrink: 0;
}
.vigil-eye-icon { width: 38px; height: 38px; flex-shrink: 0; }
.vigil-logo {
    font-family: 'Manrope', sans-serif !important;
    font-size: 1.65rem; font-weight: 900;
    letter-spacing: -0.02em; color: #0f172a; line-height: 1;
    text-decoration: none !important;
}
.vigil-logo-dot { color: #2563eb; }
.vigil-nav-links {
    display: flex; align-items: center; gap: 2rem;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem; font-weight: 500; color: #64748b;
}
.vigil-nav-links a {
    color: #64748b; text-decoration: none !important;
    transition: color 0.2s; cursor: pointer;
    background: none; border: none; padding: 0;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem; font-weight: 500;
}
.vigil-nav-links a:hover { color: #0f172a; }
.vigil-nav-links a.active { color: #2563eb; font-weight: 700; }
.nav-cta { display: flex; align-items: center; gap: 0.8rem; }
.nav-status-pill {
    display: flex; align-items: center; gap: 0.45rem;
    font-family: 'DM Mono', monospace !important; font-size: 0.72rem; font-weight: 500;
    color: #16a34a; background: #f0fdf4; border: 1px solid #bbf7d0;
    border-radius: 999px; padding: 0.32rem 0.85rem;
}
.nav-dot {
    width: 7px; height: 7px; border-radius: 50%; background: #22c55e;
    box-shadow: 0 0 5px rgba(34,197,94,0.5);
    animation: navpulse 2.5s ease-in-out infinite;
}
@keyframes navpulse { 0%,100%{opacity:1;} 50%{opacity:0.35;} }

/* Streamlit nav button container override */
.nav-streamlit-row {
    display: flex; align-items: center; gap: 2rem;
}
.nav-streamlit-row .stButton > button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #64748b !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    padding: 0 !important;
    width: auto !important;
    margin: 0 !important;
    min-height: unset !important;
    letter-spacing: 0 !important;
    transform: none !important;
    transition: color 0.2s !important;
}
.nav-streamlit-row .stButton > button:hover {
    color: #0f172a !important;
    background: transparent !important;
    transform: none !important;
    box-shadow: none !important;
}
.nav-btn-active .stButton > button {
    color: #2563eb !important;
    font-weight: 700 !important;
}

/* ═══ HERO ═══ */
.vigil-hero-wrap {
    display: flex; align-items: center; justify-content: space-between;
    gap: 2.5rem; margin-top: 1.5rem; margin-bottom: 2rem; padding: 2rem 0 1.5rem; flex-wrap: wrap;
}
.vigil-hero-text { flex:1; min-width:280px; max-width:600px; }
.vigil-hero-eyebrow {
    display: inline-flex; align-items: center; gap: 0.5rem;
    font-family: 'DM Mono', monospace !important; font-size: 0.72rem; font-weight: 500;
    letter-spacing: 0.12em; color: #2563eb; text-transform: uppercase;
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-radius: 999px; padding: 0.32rem 0.9rem; margin-bottom: 1.1rem;
}
.vigil-hero-eyebrow::before { content: '●'; font-size: 0.45rem; animation: navpulse 2s infinite; }
.vigil-hero-title {
    font-family: 'Manrope', sans-serif !important;
    font-size: clamp(2rem, 4vw, 3.4rem); font-weight: 900; color: #0f172a;
    line-height: 1.12; letter-spacing: -0.03em; margin-bottom: 0.9rem;
}
.vigil-hero-title span { color: #2563eb; }
.vigil-hero-desc {
    font-family: 'DM Sans', sans-serif !important;
    font-size: clamp(0.9rem, 1.4vw, 1rem); color: #64748b; line-height: 1.75;
}
.vigil-hero-image { flex:1; min-width:280px; max-width:620px; position:relative; }
.vigil-hero-image-inner {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 16px;
    overflow: hidden; box-shadow: 0 4px 24px rgba(0,0,0,0.08); position:relative;
}
.vigil-hero-image-inner img { width:100%; height:auto; display:block; border-radius:15px; }
.vigil-hero-img-badge {
    position:absolute; top:10px; left:10px;
    display:inline-flex; align-items:center; gap:0.4rem;
    font-family:'DM Mono',monospace !important; font-size:0.6rem; font-weight:600;
    color:#fff; background:rgba(15,23,42,0.75); border:1px solid rgba(255,255,255,0.15);
    border-radius:6px; padding:0.25rem 0.6rem; letter-spacing:0.1em; backdrop-filter:blur(4px);
}
.vigil-hero-img-badge::before {
    content:''; width:6px; height:6px; border-radius:50%;
    background:#ef4444; box-shadow:0 0 6px rgba(239,68,68,0.6);
    animation:navpulse 1.5s infinite;
}
.vigil-hero-img-placeholder {
    background:linear-gradient(135deg,#eff6ff,#e0e7ff); border-radius:15px;
    min-height:220px; display:flex; flex-direction:column;
    align-items:center; justify-content:center; gap:0.6rem; padding:2rem; text-align:center;
}
.vigil-hero-img-placeholder .ph-icon { font-size:2.5rem; opacity:0.3; margin-bottom:0.2rem; }
.vigil-hero-img-placeholder .ph-text {
    font-family:'DM Mono',monospace !important; font-size:0.7rem; color:#94a3b8; letter-spacing:0.1em;
}

/* ═══ STATUS STRIP ═══ */
.status-strip {
    display:flex; background:#fff; border:1px solid #e2e8f0;
    border-radius:14px; margin-bottom:2rem; overflow:hidden;
    flex-wrap:wrap; box-shadow:0 1px 4px rgba(0,0,0,0.05);
}
.strip-item {
    flex:1; min-width:130px; padding:1rem 1.4rem;
    border-right:1px solid #f1f5f9; transition:background 0.2s;
}
.strip-item:hover { background:#f8fafc; }
.strip-item:last-child { border-right:none; }
.strip-key {
    font-family:'DM Mono',monospace !important; font-size:0.6rem; font-weight:500;
    letter-spacing:0.12em; color:#94a3b8; text-transform:uppercase; margin-bottom:0.35rem;
}
.strip-val { font-family:'DM Sans',sans-serif !important; font-size:0.85rem; font-weight:600; color:#334155; }
.strip-val.online { color:#16a34a; }

/* ═══ CARDS ═══ */
.card {
    background:#fff; border:1px solid #e2e8f0; border-radius:16px;
    padding:1.5rem 1.7rem; position:relative; box-shadow:0 1px 6px rgba(0,0,0,0.05);
}
@media(max-width:600px){.card{padding:1.1rem;}}
.card-title {
    font-family:'DM Sans',sans-serif !important; font-size:0.7rem; font-weight:700;
    letter-spacing:0.1em; color:#94a3b8; text-transform:uppercase;
    margin-bottom:1.1rem; padding-bottom:0.85rem; border-bottom:1px solid #f1f5f9;
    display:flex; align-items:center; gap:0.55rem;
}
.card-title::before {
    content:''; display:inline-block; width:3px; height:14px;
    background:#2563eb; border-radius:2px; flex-shrink:0;
}

/* ═══ FILE UPLOADER ═══ */
[data-testid="stFileUploader"] section {
    background: #f8fafc !important;
    border: 1.5px dashed #cbd5e1 !important;
    border-radius: 12px !important;
    transition: all 0.25s !important;
    padding: 1.8rem 1.2rem !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #2563eb !important;
    background: #eff6ff !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] > div > label,
[data-testid="stFileUploader"] > label {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    overflow: hidden !important;
    padding: 0 !important;
    margin: 0 !important;
    font-size: 0 !important;
    line-height: 0 !important;
}
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span:not([data-testid]) {
    color: #64748b !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}
[data-testid="stFileUploader"] small {
    color: #94a3b8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
}
[data-testid="stFileUploader"] button {
    background: #fff !important; color: #2563eb !important;
    border: 1.5px solid #bfdbfe !important; border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.82rem !important;
    font-weight: 600 !important; padding: 0.45rem 1.1rem !important;
    box-shadow: none !important; transition: all 0.2s !important;
}
[data-testid="stFileUploader"] button:hover {
    background: #eff6ff !important; border-color: #2563eb !important;
}

/* ═══ BUTTON ═══ */
.stButton > button,
.stDownloadButton > button {
    width:100% !important; background:#2563eb !important; color:#fff !important;
    border:none !important; border-radius:10px !important;
    font-family:'DM Sans',sans-serif !important; font-size:0.96rem !important;
    font-weight:600 !important; letter-spacing:0.01em !important;
    padding:0.9rem 2rem !important; transition:all 0.2s !important;
    margin-top:0.4rem !important; box-shadow:0 4px 14px rgba(37,99,235,0.3) !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    background:#1d4ed8 !important; box-shadow:0 6px 22px rgba(37,99,235,0.4) !important;
    transform:translateY(-1px) !important;
}
.stButton > button:active,
.stDownloadButton > button:active { transform:translateY(0) !important; }

/* ═══ VIDEO ═══ */
[data-testid="stVideo"] video, video {
    border:1px solid #e2e8f0 !important; border-radius:12px !important;
    width:100% !important; background:#f8fafc !important;
    box-shadow:0 2px 8px rgba(0,0,0,0.06) !important;
}

/* ═══ GLOBE LOADER ═══ */
.globe-overlay {
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    gap:1.5rem; padding:2.5rem 1rem 2rem; background:#fff;
    border:1px solid #e2e8f0; border-radius:16px; box-shadow:0 2px 12px rgba(0,0,0,0.06);
}
.globe-wrap { position:relative; width:100px; height:100px; flex-shrink:0; }
.globe-ring { position:absolute; inset:0; border-radius:50%; border:1.5px solid rgba(37,99,235,0.15); }
.globe-arc {
    position:absolute; inset:0; border-radius:50%;
    border:2px solid transparent; border-top-color:#2563eb; border-right-color:rgba(37,99,235,0.3);
    animation:globeSpin 1.4s linear infinite; box-shadow:0 0 16px rgba(37,99,235,0.15);
}
.globe-arc2 {
    position:absolute; inset:12px; border-radius:50%;
    border:1.5px solid transparent; border-bottom-color:#60a5fa; border-left-color:rgba(96,165,250,0.3);
    animation:globeSpinR 2s linear infinite;
}
.globe-core { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; }
.globe-core-dot {
    width:16px; height:16px; border-radius:50%;
    background:radial-gradient(circle,#3b82f6,#2563eb 70%);
    box-shadow:0 0 16px rgba(37,99,235,0.5),0 0 32px rgba(37,99,235,0.15);
    animation:corePulse 1.4s ease-in-out infinite;
}
.globe-line {
    position:absolute; top:50%; left:50%; width:40px; height:1px;
    background:linear-gradient(90deg,rgba(37,99,235,0.5),transparent);
    transform-origin:left center; animation:radarSweep 2s linear infinite;
}
.globe-line:nth-child(5){animation-delay:-0.66s;opacity:0.5;}
.globe-line:nth-child(6){animation-delay:-1.33s;opacity:0.3;}
@keyframes globeSpin  { to{transform:rotate(360deg);} }
@keyframes globeSpinR { to{transform:rotate(-360deg);} }
@keyframes corePulse  { 0%,100%{box-shadow:0 0 16px rgba(37,99,235,0.5),0 0 32px rgba(37,99,235,0.15);} 50%{box-shadow:0 0 26px rgba(37,99,235,0.8),0 0 50px rgba(37,99,235,0.25);} }
@keyframes radarSweep { to{transform:rotate(360deg);} }
.globe-label {
    font-family:'DM Mono',monospace !important; font-size:0.78rem; font-weight:500;
    color:#2563eb; letter-spacing:0.12em; text-transform:uppercase; text-align:center;
}
.globe-sub {
    font-family:'DM Mono',monospace !important; font-size:0.65rem; color:#94a3b8;
    letter-spacing:0.08em; text-align:center;
    animation:textBlink 1.6s ease-in-out infinite; margin-top:0.3rem;
}
@keyframes textBlink { 0%,100%{opacity:1;} 50%{opacity:0.4;} }
.globe-steps { display:flex; flex-direction:column; gap:0.45rem; width:100%; max-width:270px; }
.globe-step {
    display:flex; align-items:center; gap:0.6rem;
    font-family:'DM Mono',monospace !important; font-size:0.7rem; color:#cbd5e1;
}
.globe-step-dot { width:7px; height:7px; border-radius:50%; background:#e2e8f0; flex-shrink:0; }
.globe-step.active .globe-step-dot { background:#2563eb; box-shadow:0 0 6px rgba(37,99,235,0.4); }
.globe-step.active { color:#475569; }
.globe-step.done .globe-step-dot { background:#22c55e; }
.globe-step.done  { color:#94a3b8; }

.analysis-done {
    display:flex; align-items:center; gap:0.7rem;
    padding:0.75rem 1.2rem; background:#f0fdf4;
    border:1px solid #bbf7d0; border-radius:10px;
    margin-bottom:0.5rem; animation:fadeUp 0.4s ease;
}
.analysis-done-icon {
    width:24px; height:24px; border-radius:50%; background:#dcfce7;
    display:flex; align-items:center; justify-content:center;
    font-size:0.75rem; color:#16a34a; flex-shrink:0;
}
.analysis-done-text { font-family:'DM Sans',sans-serif !important; font-size:0.82rem; font-weight:700; color:#15803d; }
.analysis-done-sub  { font-family:'DM Mono',monospace !important; font-size:0.63rem; color:#86efac; margin-top:0.1rem; }

/* ═══ RESULT CARD ═══ */
.result-card {
    margin-top:1.2rem; border:1px solid #e2e8f0; border-radius:16px;
    overflow:hidden; animation:fadeUp 0.4s ease; box-shadow:0 2px 12px rgba(0,0,0,0.06); background:#fff;
}
@keyframes fadeUp { from{opacity:0;transform:translateY(10px);} to{opacity:1;transform:translateY(0);} }
.result-top {
    display:flex; align-items:center; justify-content:space-between;
    padding:1rem 1.5rem; background:#f8fafc; border-bottom:1px solid #f1f5f9;
    flex-wrap:wrap; gap:0.5rem;
}
.result-top-label {
    font-family:'DM Sans',sans-serif !important; font-size:0.75rem; font-weight:700;
    color:#94a3b8; letter-spacing:0.08em; text-transform:uppercase;
}
.result-badge {
    font-family:'DM Mono',monospace !important; font-size:0.7rem; font-weight:600;
    letter-spacing:0.1em; padding:0.3rem 0.9rem; border-radius:6px; text-transform:uppercase;
}
.badge-safe   { color:#16a34a; border:1px solid #bbf7d0; background:#f0fdf4; }
.badge-fight  { color:#b45309; border:1px solid #fcd34d; background:#fffbeb; }
.badge-movie  { color:#6d28d9; border:1px solid #ddd6fe; background:#f5f3ff; }
.badge-weapon {
    color:#dc2626; border:1px solid #fca5a5; background:#fef2f2;
    animation:threatpulse 1.2s ease-in-out infinite;
}
@keyframes threatpulse { 0%,100%{box-shadow:0 0 0 0 rgba(220,38,38,0);} 50%{box-shadow:0 0 0 4px rgba(220,38,38,0.12);} }
.result-stats {
    display:grid; grid-template-columns:1fr 1fr 1fr; background:#fff;
}
@media(max-width:600px){.result-stats{grid-template-columns:1fr 1fr;}.r-stat:last-child{grid-column:span 2;border-right:none;border-top:1px solid #f1f5f9;}}
.r-stat { padding:1.2rem 1.5rem; border-right:1px solid #f1f5f9; }
.r-stat:last-child { border-right:none; }
.r-stat-label {
    font-family:'DM Sans',sans-serif !important; font-size:0.63rem; font-weight:700;
    letter-spacing:0.1em; color:#94a3b8; text-transform:uppercase; margin-bottom:0.4rem;
}
.r-stat-value { font-family:'Manrope',sans-serif !important; font-size:1.55rem; font-weight:900; color:#0f172a; line-height:1; }
.r-stat-value.c-safe    { color:#16a34a; }
.r-stat-value.c-warning { color:#b45309; }
.r-stat-value.c-moderate{ color:#6d28d9; }
.r-stat-value.c-danger  { color:#dc2626; }
.r-stat-sub { font-family:'DM Mono',monospace !important; font-size:0.62rem; color:#cbd5e1; margin-top:0.3rem; }
.conf-section { padding:1rem 1.5rem 1.3rem; background:#f8fafc; border-top:1px solid #f1f5f9; }
.conf-header {
    display:flex; justify-content:space-between;
    font-family:'DM Mono',monospace !important; font-size:0.65rem; color:#94a3b8;
    margin-bottom:0.6rem; text-transform:uppercase; letter-spacing:0.1em;
}
.conf-header span:last-child { color:#334155; font-size:0.8rem; font-weight:600; }
.conf-track { height:6px; background:#e2e8f0; border-radius:999px; overflow:hidden; }
.conf-fill  { height:100%; border-radius:999px; }

/* ═══ MODERATE ALERT ═══ */
.moderate-alert {
    display:flex; align-items:flex-start; gap:0.85rem;
    background:#f5f3ff; border:1px solid #ddd6fe;
    border-left:4px solid #7c3aed; border-radius:10px;
    padding:0.9rem 1.1rem; margin-top:1rem; animation:fadeUp 0.4s ease;
}
.moderate-alert-icon { font-size:1.1rem; flex-shrink:0; margin-top:0.05rem; }
.moderate-alert-title {
    font-family:'DM Sans',sans-serif !important; font-size:0.82rem;
    font-weight:700; color:#5b21b6; margin-bottom:0.2rem;
}
.moderate-alert-body {
    font-family:'DM Sans',sans-serif !important; font-size:0.78rem;
    color:#6d28d9; line-height:1.55;
}

/* ═══ IDLE BOXES ═══ */
.idle-box {
    background:#f8fafc; border:1.5px dashed #e2e8f0; border-radius:12px;
    min-height:160px; display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    gap:0.5rem; padding:2rem; text-align:center;
}
.idle-icon  { font-size:1.6rem; opacity:0.25; margin-bottom:0.2rem; }
.idle-title { font-family:'DM Sans',sans-serif !important; font-size:0.9rem; font-weight:600; color:#94a3b8; }
.idle-sub   { font-family:'DM Mono',monospace !important; font-size:0.67rem; color:#cbd5e1; letter-spacing:0.08em; }

/* ═══ DETECTION TABLE ═══ */
.legend-table { width:100%; border-collapse:collapse; margin-top:0.3rem; }
.legend-table tr { border-bottom:1px solid #f8fafc; transition:background 0.15s; }
.legend-table tr:last-child { border-bottom:none; }
.legend-table tr:hover { background:#f8fafc; }
.legend-table td { padding:0.82rem 0.4rem; vertical-align:middle; }
.l-dot {
    width:9px; height:9px; border-radius:50%; display:inline-block;
    margin-right:0.6rem; vertical-align:middle; flex-shrink:0;
}
.l-class { font-family:'DM Sans',sans-serif !important; font-weight:700; color:#1e293b; font-size:0.9rem; }
.l-def {
    font-family:'DM Sans',sans-serif !important; font-size:0.8rem; font-weight:400; color:#64748b;
    background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px;
    padding:0.2rem 0.55rem; display:inline-block; margin-right:0.4rem;
}
.l-badge {
    display:inline-block; font-family:'DM Mono',monospace !important;
    font-size:0.58rem; font-weight:700; padding:0.17rem 0.45rem;
    border-radius:5px; letter-spacing:0.08em; vertical-align:middle;
}

/* ═══ ERROR ═══ */
.err-box {
    border:1px solid #fca5a5; background:#fef2f2; border-radius:10px;
    padding:1.1rem 1.4rem; font-family:'DM Mono',monospace !important;
    font-size:0.78rem; color:#dc2626; line-height:1.7; margin-top:1.2rem;
}

/* ═══ MODELS PAGE ═══ */
.model-hero {
    background:linear-gradient(135deg,#eff6ff 0%,#f5f3ff 100%);
    border:1px solid #e2e8f0; border-radius:18px;
    padding:2.2rem 2.4rem; margin-bottom:1.8rem;
}
.model-hero-title {
    font-family:'Manrope',sans-serif !important; font-size:clamp(1.6rem,3vw,2.4rem);
    font-weight:900; color:#0f172a; letter-spacing:-0.02em; margin-bottom:0.5rem;
}
.model-hero-title span { color:#2563eb; }
.model-hero-sub {
    font-family:'DM Sans',sans-serif !important;
    font-size:0.95rem; color:#64748b; line-height:1.7; max-width:650px;
}
.model-card {
    background:#fff; border:1px solid #e2e8f0; border-radius:16px;
    overflow:hidden; box-shadow:0 2px 10px rgba(0,0,0,0.05); margin-bottom:1.6rem;
}
.model-card-header {
    display:flex; align-items:center; gap:1rem;
    padding:1.3rem 1.6rem; border-bottom:1px solid #f1f5f9;
    background:linear-gradient(90deg,#f8fafc,#ffffff);
}
.model-card-badge {
    display:inline-flex; align-items:center; justify-content:center;
    width:44px; height:44px; border-radius:12px; flex-shrink:0;
    font-size:1.3rem;
}
.model-card-name {
    font-family:'Manrope',sans-serif !important; font-size:1.25rem;
    font-weight:800; color:#0f172a; letter-spacing:-0.01em;
}
.model-card-tag {
    font-family:'DM Mono',monospace !important; font-size:0.65rem; font-weight:500;
    color:#2563eb; background:#eff6ff; border:1px solid #bfdbfe;
    border-radius:4px; padding:0.18rem 0.55rem; letter-spacing:0.08em; text-transform:uppercase;
    margin-left:0.5rem; vertical-align:middle;
}
.model-card-body { padding:1.4rem 1.6rem; }
.model-section-title {
    font-family:'DM Sans',sans-serif !important; font-size:0.65rem; font-weight:700;
    letter-spacing:0.14em; color:#94a3b8; text-transform:uppercase;
    margin-bottom:0.55rem; display:flex; align-items:center; gap:0.5rem;
}
.model-section-title::after { content:''; flex:1; height:1px; background:#f1f5f9; }
.model-para {
    font-family:'DM Sans',sans-serif !important;
    font-size:0.9rem !important; color:#475569 !important;
    line-height:1.75 !important; margin-bottom:1.2rem !important;
    font-weight:400 !important;
}
.model-para strong, .model-para b {
    font-family:'DM Sans',sans-serif !important;
    font-weight:700 !important; color:#1e293b !important;
}
.model-arch-grid {
    display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
    gap:0.75rem; margin-bottom:1.2rem;
}
.model-arch-item {
    background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
    padding:0.85rem 1rem;
}
.model-arch-key {
    font-family:'DM Mono',monospace !important; font-size:0.58rem; font-weight:500;
    letter-spacing:0.1em; color:#94a3b8; text-transform:uppercase; margin-bottom:0.3rem;
}
.model-arch-val {
    font-family:'DM Sans',sans-serif !important; font-size:0.85rem; font-weight:600; color:#1e293b;
}
.model-ft-step {
    display:flex; gap:0.75rem; padding:0.75rem 0;
    border-bottom:1px solid #f8fafc;
}
.model-ft-step:last-child { border-bottom:none; }
.model-ft-num {
    width:24px; height:24px; border-radius:50%; background:#eff6ff; border:1px solid #bfdbfe;
    display:flex; align-items:center; justify-content:center; flex-shrink:0;
    font-family:'DM Mono',monospace !important; font-size:0.65rem; font-weight:700; color:#2563eb;
}
.model-ft-text {
    font-family:'DM Sans',sans-serif !important;
    font-size:0.87rem !important; color:#475569 !important; line-height:1.6 !important;
    font-weight:400 !important;
}
.model-ft-text strong, .model-ft-text b {
    font-family:'DM Sans',sans-serif !important;
    font-weight:700 !important; color:#0f172a !important;
}

/* ═══ REPORTS PAGE ═══ */
.report-empty {
    background:#fff; border:1.5px dashed #e2e8f0; border-radius:16px;
    min-height:280px; display:flex; flex-direction:column;
    align-items:center; justify-content:center; gap:0.75rem; text-align:center; padding:2.5rem;
}
.report-empty-icon { font-size:2.2rem; opacity:0.2; }
.report-empty-title {
    font-family:'DM Sans',sans-serif !important; font-size:1rem; font-weight:700; color:#94a3b8;
}
.report-empty-sub {
    font-family:'DM Mono',monospace !important; font-size:0.68rem; color:#cbd5e1; letter-spacing:0.08em;
}
.report-meta-grid {
    display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
    gap:0.75rem; margin:1.2rem 0;
}
.report-meta-item {
    background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:0.9rem 1rem;
}
.report-meta-key {
    font-family:'DM Mono',monospace !important; font-size:0.58rem; font-weight:500;
    letter-spacing:0.1em; color:#94a3b8; text-transform:uppercase; margin-bottom:0.3rem;
}
.report-meta-val {
    font-family:'DM Sans',sans-serif !important; font-size:0.9rem; font-weight:700; color:#1e293b;
}

/* ═══ SETTINGS PAGE ═══ */
.settings-section-title {
    font-family:'Manrope',sans-serif !important; font-size:1.4rem; font-weight:800;
    color:#0f172a; letter-spacing:-0.01em; margin-bottom:0.35rem;
}
.settings-section-sub {
    font-family:'DM Sans',sans-serif !important;
    font-size:0.88rem; color:#64748b; margin-bottom:1.5rem;
}
.support-card {
    background:#fff; border:1px solid #e2e8f0; border-radius:16px;
    overflow:hidden; box-shadow:0 2px 10px rgba(0,0,0,0.05); margin-bottom:1.2rem;
}
.support-card-header {
    padding:1rem 1.4rem; border-bottom:1px solid #f1f5f9; background:#f8fafc;
    font-family:'DM Sans',sans-serif !important; font-size:0.7rem; font-weight:700;
    letter-spacing:0.1em; color:#94a3b8; text-transform:uppercase;
    display:flex; align-items:center; gap:0.5rem;
}
.support-card-header::before {
    content:''; width:3px; height:14px; background:#2563eb; border-radius:2px; flex-shrink:0;
}
.support-card-body { padding:1.4rem; }
.support-channel {
    display:flex; align-items:center; gap:0.9rem;
    padding:0.85rem 0; border-bottom:1px solid #f8fafc;
}
.support-channel:last-child { border-bottom:none; }
.support-channel-icon {
    width:38px; height:38px; border-radius:10px; display:flex;
    align-items:center; justify-content:center; font-size:1.1rem; flex-shrink:0;
}
.support-channel-title {
    font-family:'DM Sans',sans-serif !important; font-size:0.88rem; font-weight:700; color:#1e293b;
}
.support-channel-sub {
    font-family:'DM Mono',monospace !important; font-size:0.65rem; color:#94a3b8;
    letter-spacing:0.05em; margin-top:0.1rem;
}
.support-channel-link {
    margin-left:auto; font-family:'DM Sans',sans-serif !important; font-size:0.78rem;
    font-weight:600; color:#2563eb; text-decoration:none;
    background:#eff6ff; border:1px solid #bfdbfe; border-radius:7px;
    padding:0.35rem 0.85rem; white-space:nowrap; transition:all 0.2s;
}
.support-channel-link:hover { background:#dbeafe; color:#1d4ed8; }
.faq-item { padding:0.9rem 0; border-bottom:1px solid #f8fafc; }
.faq-item:last-child { border-bottom:none; }
.faq-q {
    font-family:'DM Sans',sans-serif !important; font-size:0.88rem;
    font-weight:700; color:#1e293b; margin-bottom:0.35rem;
}
.faq-a {
    font-family:'DM Sans',sans-serif !important;
    font-size:0.84rem !important; color:#64748b !important; line-height:1.65 !important;
}
.faq-a code {
    font-family:'DM Mono',monospace !important;
    background:#f1f5f9; padding:0.1rem 0.4rem; border-radius:4px;
    font-size:0.82rem !important;
}

/* ═══ HELP BAR ═══ */
.help-bar {
    display:flex; align-items:center; justify-content:space-between;
    background:#fff; border:1px solid #e2e8f0; border-radius:14px;
    padding:1rem 1.5rem; margin-top:1.5rem; box-shadow:0 1px 4px rgba(0,0,0,0.04);
    flex-wrap:wrap; gap:1rem;
}
.help-bar-left { display:flex; align-items:center; gap:0.75rem; }
.help-bar-icon {
    width:36px; height:36px; border-radius:9px; background:#eff6ff;
    border:1px solid #bfdbfe; display:flex; align-items:center; justify-content:center;
    font-size:1rem; flex-shrink:0;
}
.help-bar-title {
    font-family:'DM Sans',sans-serif !important; font-size:0.9rem; font-weight:700; color:#1e293b;
}
.help-bar-sub {
    font-family:'DM Mono',monospace !important; font-size:0.63rem; color:#94a3b8;
    letter-spacing:0.07em; margin-top:0.1rem;
}
.help-bar-btn {
    display:inline-flex; align-items:center; gap:0.45rem; background:#2563eb;
    color:#fff !important; text-decoration:none !important;
    font-family:'DM Sans',sans-serif !important; font-size:0.82rem; font-weight:600;
    padding:0.6rem 1.3rem; border-radius:9px; border:none; cursor:pointer;
    transition:all 0.2s; box-shadow:0 3px 12px rgba(37,99,235,0.25); white-space:nowrap; flex-shrink:0;
}
.help-bar-btn:hover { background:#1d4ed8 !important; box-shadow:0 5px 18px rgba(37,99,235,0.35); transform:translateY(-1px); color:#fff !important; }
.help-bar-btn svg { width:14px; height:14px; flex-shrink:0; }

/* ═══ FOOTER ═══ */
.vigil-divider { height:1px; background:#e2e8f0; margin:1.5rem 0 0; }
.vigil-footer {
    display:flex; justify-content:space-between; align-items:center;
    font-family:'DM Mono',monospace !important; font-size:0.65rem; color:#cbd5e1; letter-spacing:0.1em;
    padding:0.9rem 0 0.5rem; flex-wrap:wrap; gap:0.4rem;
}

/* ═══ CCTV BADGE STRIP ═══ */
.cctv-badge-strip { display:flex; align-items:center; gap:0.5rem; margin-bottom:1rem; flex-wrap:wrap; }
.cctv-badge {
    display:inline-flex; align-items:center; gap:0.4rem;
    font-family:'DM Mono',monospace !important; font-size:0.63rem; font-weight:500; color:#64748b;
    background:#fff; border:1px solid #e2e8f0; border-radius:6px;
    padding:0.28rem 0.65rem; letter-spacing:0.07em;
}

/* ═══ PAGE HEADER ═══ */
.page-header {
    margin-bottom:2rem; padding-bottom:1.5rem; border-bottom:1px solid #e2e8f0;
}
.page-header-eyebrow {
    font-family:'DM Mono',monospace !important; font-size:0.65rem; font-weight:500;
    letter-spacing:0.12em; color:#2563eb; text-transform:uppercase; margin-bottom:0.4rem;
}
.page-header-title {
    font-family:'Manrope',sans-serif !important; font-size:clamp(1.6rem,3vw,2.2rem);
    font-weight:900; color:#0f172a; letter-spacing:-0.02em; line-height:1.15;
}
.page-header-sub {
    font-family:'DM Sans',sans-serif !important;
    font-size:0.92rem; color:#64748b; margin-top:0.45rem; line-height:1.65;
}

@media(max-width:768px){
    [data-testid="column"] { min-width:100% !important; }
    .stHorizontalBlock { flex-wrap:wrap !important; }
    .vigil-hero-wrap { flex-direction:column; }
    .vigil-hero-image { max-width:100%; }
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  NAVIGATION — using Streamlit native buttons styled as links
#  Order: Dashboard | Reports | Models | Settings
# ═══════════════════════════════════════════════════════

# Logo + left side with nav links
st.markdown(f"""
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
        <a href="#dashboard">Dashboard</a>
        <a href="#reports">Reports</a>
        <a href="#models">Models</a>
        <a href="#settings">Settings</a>
    </div>
    <div class="nav-status-pill"><span class="nav-dot"></span>System Online</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  SECTION: DASHBOARD
# ═══════════════════════════════════════════════════════
md('<section id="dashboard" class="vigil-section">')

# ── HERO ──────────────────────────────────────────
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
        '</div></div></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="vigil-hero-wrap">' + HERO_TEXT + """
        <div class="vigil-hero-image">
            <div class="vigil-hero-image-inner">
                <div class="vigil-hero-img-placeholder">
                    <div class="ph-icon">&#127909;</div>
                    <div class="ph-text">PLACE demo.png IN images/ FOLDER</div>
                </div>
            </div>
        </div>
        </div>""",
        unsafe_allow_html=True,
    )

# ── CCTV BADGE STRIP ────────────────────────────────
st.markdown("""
<div class="cctv-badge-strip">
    <span class="cctv-badge">&#128249; R3D-18 &middot; 4-Class</span>
    <span class="cctv-badge">&#128269; YOLOv8 &middot; Weapon Detector</span>
    <span class="cctv-badge">&#9889; FastAPI &middot; 127.0.0.1:8000</span>
    <span class="cctv-badge">&#127902; 5-Frame Majority Vote</span>
    <span class="cctv-badge">&#128336; Real-Time Inference</span>
</div>
""", unsafe_allow_html=True)

# ── STATUS STRIP ────────────────────────────────────
st.markdown("""
<div class="status-strip">
    <div class="strip-item"><div class="strip-key">Model Status</div><div class="strip-val online">&#9679; Loaded &amp; Ready</div></div>
    <div class="strip-item"><div class="strip-key">Violence Classifier</div><div class="strip-val">R3D-18 &middot; 4-Class</div></div>
    <div class="strip-item"><div class="strip-key">Weapon Detector</div><div class="strip-val">YOLOv8 &middot; 4-Class</div></div>
    <div class="strip-item"><div class="strip-key">Inference Backend</div><div class="strip-val">FastAPI &middot; 127.0.0.1:8000</div></div>
    <div class="strip-item"><div class="strip-key">Smoothing Window</div><div class="strip-val">5 Frames &middot; Majority Vote</div></div>
</div>
""", unsafe_allow_html=True)

# ── TWO COLUMNS ─────────────────────────────────────
col_left, col_right = st.columns([1.05, 1], gap="large")

with col_left:
    st.markdown('<div class="card" style="margin-bottom:1.4rem"><div class="card-title">Input — Surveillance Feed</div>', unsafe_allow_html=True)
    st.markdown('<div class="uploader-label" style="margin-bottom:0.75rem;font-weight:600;color:#0f172a;">Upload Surveillance Feed</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="Upload Surveillance Feed",
        type=["mp4"],
        label_visibility="hidden"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        st.session_state.analysis_filename = uploaded_file.name
        st.markdown('<div class="card" style="margin-bottom:1.4rem"><div class="card-title">Preview — Source Feed</div>', unsafe_allow_html=True)
        st.video(uploaded_file)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="card-title">Detection Classes</div>
        <table class="legend-table">
            <tr>
                <td style="width:36%"><span class="l-dot" style="background:#22c55e;box-shadow:0 0 5px rgba(34,197,94,0.4)"></span><span class="l-class">NonFight</span></td>
                <td><span class="l-def">No violent activity detected</span><span class="l-badge" style="color:#16a34a;border:1px solid #bbf7d0;background:#f0fdf4">SAFE</span></td>
            </tr>
            <tr>
                <td><span class="l-dot" style="background:#f59e0b;box-shadow:0 0 5px rgba(245,158,11,0.4)"></span><span class="l-class">Fight</span></td>
                <td><span class="l-def">Physical altercation between subjects</span><span class="l-badge" style="color:#b45309;border:1px solid #fcd34d;background:#fffbeb">HIGH</span></td>
            </tr>
            <tr>
                <td><span class="l-dot" style="background:#fb923c;box-shadow:0 0 5px rgba(251,146,60,0.4)"></span><span class="l-class">HockeyFight</span></td>
                <td><span class="l-def">Sport-context violent confrontation</span><span class="l-badge" style="color:#c2410c;border:1px solid #fed7aa;background:#fff7ed">HIGH</span></td>
            </tr>
            <tr>
                <td><span class="l-dot" style="background:#a78bfa;box-shadow:0 0 5px rgba(167,139,250,0.4)"></span><span class="l-class">MovieFight</span></td>
                <td><span class="l-def">Scripted / cinematic fight sequence</span><span class="l-badge" style="color:#7c3aed;border:1px solid #ddd6fe;background:#f5f3ff">MODERATE</span></td>
            </tr>
            <tr>
                <td><span class="l-dot" style="background:#ef4444;box-shadow:0 0 5px rgba(239,68,68,0.4)"></span><span class="l-class">Weaponized</span></td>
                <td><span class="l-def">Knife &middot; Handgun &middot; Rifle &middot; Launcher</span><span class="l-badge" style="color:#dc2626;border:1px solid #fca5a5;background:#fef2f2">CRITICAL</span></td>
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
                <div class="globe-step done"><div class="globe-step-dot"></div>Video decoded &amp; cleaned</div>
                <div class="globe-step active"><div class="globe-step-dot"></div>R3D-18 violence classification</div>
                <div class="globe-step active"><div class="globe-step-dot"></div>YOLOv8 weapon detection</div>
                <div class="globe-step"><div class="globe-step-dot"></div>Frame annotation &amp; export</div>
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
            response  = None
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

            st.session_state.analysis_result    = result
            st.session_state.analysis_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if "weapon" in pred_lower:
                badge_cls = "badge-weapon"; val_cls = "c-danger"
                bar_color = "#dc2626"; threat_lv = "CRITICAL"; threat_col = "#dc2626"
                show_movie_alert = False
            elif "movie" in pred_lower:
                badge_cls = "badge-movie"; val_cls = "c-moderate"
                bar_color = "#7c3aed"; threat_lv = "MODERATE"; threat_col = "#6d28d9"
                show_movie_alert = True
            elif "fight" in pred_lower:
                badge_cls = "badge-fight"; val_cls = "c-warning"
                bar_color = "#f59e0b"; threat_lv = "HIGH"; threat_col = "#b45309"
                show_movie_alert = False
            else:
                badge_cls = "badge-safe"; val_cls = "c-safe"
                bar_color = "#22c55e"; threat_lv = "NONE"; threat_col = "#16a34a"
                show_movie_alert = False

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

            if show_movie_alert:
                st.markdown("""
                <div class="moderate-alert">
                    <div class="moderate-alert-icon">&#9432;</div>
                    <div>
                        <div class="moderate-alert-title">Moderate Threat — Scripted/Cinematic Content Detected</div>
                        <div class="moderate-alert-body">
                            The model identified patterns consistent with a <strong>scripted or movie-style fight sequence</strong>.
                            This classification typically indicates choreographed action rather than a real-world altercation.
                            Real-world threat probability is <strong>low to moderate</strong>. No immediate intervention is recommended,
                            but the feed should be reviewed in context.
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

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
                    st.session_state.analysis_video = browser_bytes
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

            # Link to Reports page
            st.markdown("""
            <div style="margin-top:1rem;text-align:center;">
                <a href="#reports"
                   style="display:inline-flex;align-items:center;gap:0.45rem;
                          font-family:'DM Sans',sans-serif;font-size:0.8rem;font-weight:600;
                          color:#2563eb;background:#eff6ff;border:1px solid #bfdbfe;
                          padding:0.5rem 1.1rem;border-radius:8px;text-decoration:none;">
                    &#128196; Download PDF Report
                </a>
            </div>
            """, unsafe_allow_html=True)

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

md('</section>')
md('<div class="vigil-section-spacer"></div>')

# ═══════════════════════════════════════════════════════
#  SECTION: REPORTS
# ═══════════════════════════════════════════════════════
md('<section id="reports" class="vigil-section">')

st.markdown("""
<div class="page-header">
    <div class="page-header-eyebrow">Analysis Output</div>
    <div class="page-header-title">Reports</div>
    <div class="page-header-sub">
        Download a detailed PDF report for any completed threat analysis.
        Run an analysis on the Dashboard first, then return here to generate the report.
    </div>
</div>
""", unsafe_allow_html=True)

result = st.session_state.analysis_result

if result is None:
    st.markdown("""
    <div class="report-empty">
        <div class="report-empty-icon">&#128196;</div>
        <div class="report-empty-title">No Analysis Found</div>
        <div class="report-empty-sub">
            GO TO DASHBOARD &nbsp;&middot;&nbsp; UPLOAD A VIDEO &nbsp;&middot;&nbsp; RUN THREAT ANALYSIS
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div style="margin-top:1.2rem"></div>', unsafe_allow_html=True)
    st.markdown("""
    <a href="#dashboard"
       style="display:inline-flex;align-items:center;gap:0.45rem;background:#2563eb;
              color:#fff;text-decoration:none;font-family:'DM Sans',sans-serif;
              font-size:0.85rem;font-weight:600;padding:0.65rem 1.4rem;
              border-radius:9px;box-shadow:0 3px 12px rgba(37,99,235,0.25);">
        &#8592; Go to Dashboard
    </a>
    """, unsafe_allow_html=True)

else:
    pred       = result["prediction"]
    conf       = result["confidence"]
    filename   = st.session_state.analysis_filename
    timestamp  = st.session_state.analysis_timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pred_lower = pred.lower()

    if "weapon" in pred_lower:
        threat_lv  = "CRITICAL"; thr_color = "#dc2626"
    elif "movie" in pred_lower:
        threat_lv  = "MODERATE"; thr_color = "#6d28d9"
    elif "fight" in pred_lower:
        threat_lv  = "HIGH"; thr_color = "#b45309"
    else:
        threat_lv  = "NONE"; thr_color = "#16a34a"

    st.markdown(f"""
    <div class="card">
        <div class="card-title">Latest Analysis Report</div>
        <div class="report-meta-grid">
            <div class="report-meta-item">
                <div class="report-meta-key">File</div>
                <div class="report-meta-val" style="font-size:0.8rem;word-break:break-all;">{filename}</div>
            </div>
            <div class="report-meta-item">
                <div class="report-meta-key">Classification</div>
                <div class="report-meta-val">{pred}</div>
            </div>
            <div class="report-meta-item">
                <div class="report-meta-key">Confidence</div>
                <div class="report-meta-val">{conf}%</div>
            </div>
            <div class="report-meta-item">
                <div class="report-meta-key">Threat Level</div>
                <div class="report-meta-val" style="color:{thr_color};">{threat_lv}</div>
            </div>
            <div class="report-meta-item">
                <div class="report-meta-key">Timestamp</div>
                <div class="report-meta-val" style="font-size:0.8rem;">{timestamp}</div>
            </div>
            <div class="report-meta-item">
                <div class="report-meta-key">Pipeline</div>
                <div class="report-meta-val" style="font-size:0.75rem;">R3D-18 + YOLOv8</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)

    pdf_bytes = generate_pdf_report(filename, pred, conf, threat_lv, timestamp)
    is_pdf = pdf_bytes[:4] == b'%PDF'
    ext    = "pdf" if is_pdf else "txt"
    mime   = "application/pdf" if is_pdf else "text/plain"
    label  = "⬇ Download PDF Report" if is_pdf else "⬇ Download Report (install fpdf2 for PDF)"

    st.download_button(
        label=label,
        data=pdf_bytes,
        file_name=f"VIGILAI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}",
        mime=mime,
        use_container_width=True,
    )

    if not is_pdf:
        st.markdown("""
        <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#94a3b8;
                    margin-top:0.6rem;padding:0.7rem 1rem;background:#f8fafc;
                    border:1px solid #e2e8f0;border-radius:8px;">
            &#9432; &nbsp;Install <strong>fpdf2</strong> for full PDF reports:
            <code style="background:#e2e8f0;padding:0.1rem 0.4rem;border-radius:4px;font-family:'DM Mono',monospace;">pip install fpdf2</code>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.analysis_video is not None:
        st.markdown('<div class="card" style="margin-top:1.4rem"><div class="card-title">Annotated Output Feed</div>', unsafe_allow_html=True)
        st.video(st.session_state.analysis_video)
        st.markdown("</div>", unsafe_allow_html=True)

md('</section>')
md('<div class="vigil-section-spacer"></div>')

# ═══════════════════════════════════════════════════════
#  SECTION: MODELS
# ═══════════════════════════════════════════════════════
md('<section id="models" class="vigil-section">')

md("""
<div class="page-header">
    <div class="page-header-eyebrow">Inference Pipeline</div>
    <div class="page-header-title">Detection Models</div>
    <div class="page-header-sub">
        Deep-learning models powering the VIGIL.AI pipeline — architecture details,
        training datasets, and fine-tuning methodology.
    </div>
</div>
""")

# ── R3D-18 ──────────────────────────────────────────
md("""
<div class="model-card">
    <div class="model-card-header">
        <div class="model-card-badge" style="background:#eff6ff;border:1px solid #bfdbfe;">&#127910;</div>
        <div>
            <span class="model-card-name">R3D-18</span>
            <span class="model-card-tag">Violence Classifier</span>
            <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#94a3b8;margin-top:0.25rem;letter-spacing:0.07em;">
                3D Residual Network &nbsp;&middot;&nbsp; 4-Class &nbsp;&middot;&nbsp; 16-Frame Clips
            </div>
        </div>
    </div>
    <div class="model-card-body">

        <div class="model-section-title">Introduction</div>
        <p class="model-para">
            R3D-18 is a <strong>3D Residual Network</strong> (He et al., 2015 + Tran et al., 2018)
            designed for video understanding. Unlike 2D CNNs that process frames in isolation,
            R3D-18 applies 3D convolutions across both spatial and temporal dimensions simultaneously,
            enabling the model to learn motion patterns, interaction dynamics, and action sequences
            directly from raw video clips — making it highly effective for violence detection in
            surveillance footage.
        </p>

        <div class="model-section-title">Architecture</div>
        <div class="model-arch-grid">
            <div class="model-arch-item">
                <div class="model-arch-key">Input Shape</div>
                <div class="model-arch-val">3 &times; 16 &times; 112 &times; 112</div>
            </div>
            <div class="model-arch-item">
                <div class="model-arch-key">Backbone</div>
                <div class="model-arch-val">ResNet-18 (3D Conv)</div>
            </div>
            <div class="model-arch-item">
                <div class="model-arch-key">Temporal Depth</div>
                <div class="model-arch-val">16 frames / clip</div>
            </div>
            <div class="model-arch-item">
                <div class="model-arch-key">Residual Blocks</div>
                <div class="model-arch-val">4 &times; [2 blocks each]</div>
            </div>
            <div class="model-arch-item">
                <div class="model-arch-key">Output Classes</div>
                <div class="model-arch-val">4 (Softmax)</div>
            </div>
            <div class="model-arch-item">
                <div class="model-arch-key">Smoothing</div>
                <div class="model-arch-val">5-Frame Majority Vote</div>
            </div>
        </div>
        <p class="model-para">
            The network stacks four residual stages, each containing two 3D convolutional blocks
            with shortcut connections. A global average pooling layer reduces the spatial-temporal
            feature map to a 512-dimensional vector, which is passed to a fully-connected head
            outputting 4 logits. Softmax produces per-class probabilities; temporal stability is
            achieved by aggregating predictions across a sliding 5-frame window via majority vote.
        </p>

        <div class="model-section-title">Fine-Tuning Process</div>
        <div class="model-ft-step">
            <div class="model-ft-num">1</div>
            <div class="model-ft-text">
                <strong>Pre-training:</strong> Initialised from Kinetics-400 weights — a large-scale
                action recognition dataset with 400 action categories — providing a rich
                spatiotemporal feature prior before domain-specific fine-tuning.
            </div>
        </div>
        <div class="model-ft-step">
            <div class="model-ft-num">2</div>
            <div class="model-ft-text">
                <strong>Dataset:</strong> Fine-tuned on a curated 4-class violence corpus covering
                Fight, HockeyFight, MovieFight, and NonFight categories, sourced from public benchmark
                datasets and augmented with custom surveillance clips. Class imbalance was addressed
                via weighted random sampling during training.
            </div>
        </div>
        <div class="model-ft-step">
            <div class="model-ft-num">3</div>
            <div class="model-ft-text">
                <strong>Training regime:</strong> All layers unfrozen after an initial warm-up phase
                (top FC only). Optimised with AdamW (lr = 1e-4, weight decay = 1e-4),
                cosine annealing LR schedule, and mixed-precision (FP16) over 30 epochs on an
                NVIDIA RTX GPU.
            </div>
        </div>
        <div class="model-ft-step">
            <div class="model-ft-num">4</div>
            <div class="model-ft-text">
                <strong>Augmentation:</strong> Random horizontal flip, random crop (112 x 112),
                temporal jitter (random start frame within clip), colour jitter, and normalisation
                to ImageNet mean and standard deviation values.
            </div>
        </div>
        <div class="model-ft-step">
            <div class="model-ft-num">5</div>
            <div class="model-ft-text">
                <strong>Evaluation:</strong> Best checkpoint selected on validation accuracy.
                Final model exported via TorchScript for efficient inference in the FastAPI backend.
            </div>
        </div>
    </div>
</div>
""")

# ── YOLOv8 ──────────────────────────────────────────
md("""
<div class="model-card">
    <div class="model-card-header">
        <div class="model-card-badge" style="background:#fef3c7;border:1px solid #fcd34d;">&#128270;</div>
        <div>
            <span class="model-card-name">YOLOv8</span>
            <span class="model-card-tag" style="color:#b45309;background:#fffbeb;border-color:#fcd34d;">Weapon Detector</span>
            <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#94a3b8;margin-top:0.25rem;letter-spacing:0.07em;">
                Anchor-Free Object Detector &nbsp;&middot;&nbsp; 4-Class Weapons &nbsp;&middot;&nbsp; Frame-Level
            </div>
        </div>
    </div>
    <div class="model-card-body">

        <div class="model-section-title">Introduction</div>
        <p class="model-para">
            YOLOv8 (Ultralytics, 2023) is the eighth generation of the You Only Look Once
            real-time object detection family. It adopts an <strong>anchor-free detection head</strong>,
            decoupled classification and localisation branches, and a C2f backbone block that
            improves gradient flow throughout the network. Applied frame-by-frame to surveillance
            video, it provides precise bounding-box localisation of weapons — enabling the pipeline
            to override the violence classification with a Weaponized alert whenever a weapon is found.
        </p>

        <div class="model-section-title">Architecture</div>
        <div class="model-arch-grid">
            <div class="model-arch-item">
                <div class="model-arch-key">Variant</div>
                <div class="model-arch-val">YOLOv8n / YOLOv8s</div>
            </div>
            <div class="model-arch-item">
                <div class="model-arch-key">Input Resolution</div>
                <div class="model-arch-val">640 &times; 640 px</div>
            </div>
            <div class="model-arch-item">
                <div class="model-arch-key">Backbone</div>
                <div class="model-arch-val">CSPDarknet + C2f</div>
            </div>
            <div class="model-arch-item">
                <div class="model-arch-key">Neck</div>
                <div class="model-arch-val">PANet (multi-scale)</div>
            </div>
            <div class="model-arch-item">
                <div class="model-arch-key">Output Classes</div>
                <div class="model-arch-val">4 (weapon types)</div>
            </div>
            <div class="model-arch-item">
                <div class="model-arch-key">Detection Mode</div>
                <div class="model-arch-val">Anchor-Free</div>
            </div>
        </div>
        <p class="model-para">
            The backbone extracts multi-scale feature maps at three pyramid levels (P3, P4, P5).
            These are fused by the Path Aggregation Network neck and fed to three decoupled detection
            heads — one per scale — each predicting bounding boxes (centre, width, height) and
            class probabilities independently. Non-maximum suppression is applied post-inference
            to remove overlapping detections. The four weapon classes detected are:
            <strong>Knife, Handgun, Rifle,</strong> and <strong>Launcher</strong>.
        </p>

        <div class="model-section-title">Fine-Tuning Process</div>
        <div class="model-ft-step">
            <div class="model-ft-num">1</div>
            <div class="model-ft-text">
                <strong>Pre-training:</strong> Initialised from COCO pre-trained weights (80-class
                general object detection), providing strong low-level feature representations
                for the shape and edge detection that is critical for weapon recognition.
            </div>
        </div>
        <div class="model-ft-step">
            <div class="model-ft-num">2</div>
            <div class="model-ft-text">
                <strong>Dataset:</strong> Fine-tuned on a combined weapon detection dataset of
                annotated frames containing knives, handguns, rifles, and launchers drawn from
                public datasets (Open Images, Gun Detection datasets) and custom-labelled
                surveillance frames. Labels stored in YOLO format (.txt per image).
            </div>
        </div>
        <div class="model-ft-step">
            <div class="model-ft-num">3</div>
            <div class="model-ft-text">
                <strong>Training regime:</strong> 50 epochs, SGD optimiser (lr = 0.01, momentum = 0.937),
                linear LR warm-up for the first 3 epochs, mosaic augmentation throughout, and
                mixed-precision training. Early stopping triggered on validation mAP50.
            </div>
        </div>
        <div class="model-ft-step">
            <div class="model-ft-num">4</div>
            <div class="model-ft-text">
                <strong>Augmentation:</strong> Mosaic (4-image composite), random horizontal flip,
                scale, translate and shear jitter, HSV colour augmentation, and copy-paste
                augmentation to improve small-object detection.
            </div>
        </div>
        <div class="model-ft-step">
            <div class="model-ft-num">5</div>
            <div class="model-ft-text">
                <strong>Inference integration:</strong> Frames extracted from the uploaded video
                are passed individually to YOLOv8. Any frame with a weapon detection at or above
                the confidence threshold triggers a Weaponized classification override, and bounding
                boxes are drawn on the annotated output video returned to the user.
            </div>
        </div>
    </div>
</div>
""")

md('</section>')
md('<div class="vigil-section-spacer"></div>')

# ═══════════════════════════════════════════════════════
#  SECTION: SETTINGS
# ═══════════════════════════════════════════════════════
md('<section id="settings" class="vigil-section">')

st.markdown("""
<div class="page-header">
    <div class="page-header-eyebrow">Configuration &amp; Help</div>
    <div class="page-header-title">Settings &amp; Support</div>
    <div class="page-header-sub">
        Get help, report issues, or request new features for the VIGIL.AI platform.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Customer Support — top of page ──────────────────
st.markdown("""
<div class="support-card">
    <div class="support-card-header">Customer Support</div>
    <div class="support-card-body">
        <div class="support-channel">
            <div class="support-channel-icon" style="background:#eff6ff;border:1px solid #bfdbfe;">&#128139;</div>
            <div>
                <div class="support-channel-title">Email Support</div>
                <div class="support-channel-sub">kumarashisha54@gmail.com &nbsp;&middot;&nbsp; Response within 24 hrs</div>
            </div>
            <a class="support-channel-link"
               href="mailto:kumarashisha54@gmail.com?subject=VIGIL.AI%20Support%20Request&body=Hi%2C%0A%0AI%20need%20help%20with%20VIGIL.AI%3A%0A%0A"
               target="_blank">Send Email</a>
        </div>
        <div class="support-channel">
            <div class="support-channel-icon" style="background:#f0fdf4;border:1px solid #bbf7d0;">&#128172;</div>
            <div>
                <div class="support-channel-title">Bug Reports</div>
                <div class="support-channel-sub">Describe the issue &amp; steps to reproduce</div>
            </div>
            <a class="support-channel-link"
               href="mailto:kumarashisha54@gmail.com?subject=VIGIL.AI%20Bug%20Report&body=Bug%20Description%3A%0A%0ASteps%20to%20Reproduce%3A%0A1.%0A2.%0A3.%0A%0AExpected%20Behaviour%3A%0A%0AActual%20Behaviour%3A%0A"
               target="_blank">Report Bug</a>
        </div>
        <div class="support-channel">
            <div class="support-channel-icon" style="background:#fefce8;border:1px solid #fde68a;">&#128161;</div>
            <div>
                <div class="support-channel-title">Feature Requests</div>
                <div class="support-channel-sub">Suggest improvements or new capabilities</div>
            </div>
            <a class="support-channel-link"
               href="mailto:kumarashisha54@gmail.com?subject=VIGIL.AI%20Feature%20Request&body=Feature%20Description%3A%0A%0AUse%20Case%3A%0A%0APriority%20(Low%2FMedium%2FHigh)%3A%0A"
               target="_blank">Request Feature</a>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="support-card" style="margin-top:1.2rem">
    <div class="support-card-header">Frequently Asked Questions</div>
    <div class="support-card-body">
        <div class="faq-item">
            <div class="faq-q">What video formats are supported?</div>
            <div class="faq-a">Only MP4 files are currently supported, up to 200 MB per file.
                Ensure the video is encoded in H.264 for best compatibility. The pipeline will
                automatically re-encode the annotated output for browser playback via ffmpeg.</div>
        </div>
        <div class="faq-item">
            <div class="faq-q">Why is the upload button shown but the backend fails?</div>
            <div class="faq-a">The FastAPI backend must be running at
                <code>127.0.0.1:8000</code> before submitting a video. Start it with
                <code>uvicorn main:app --reload</code> in your project directory.</div>
        </div>
        <div class="faq-item">
            <div class="faq-q">What does "MovieFight — MODERATE" mean?</div>
            <div class="faq-a">The R3D-18 model detected visual patterns consistent with scripted
                or cinematic fight sequences (e.g. choreographed action from movies or TV). This class
                carries a lower real-world threat probability than genuine fight or weapon detections,
                hence the MODERATE risk level rather than HIGH.</div>
        </div>
        <div class="faq-item">
            <div class="faq-q">How do I download the analysis report?</div>
            <div class="faq-a">After running a threat analysis on the Dashboard, navigate to the
                Reports tab. You will find a Download PDF Report button with all analysis metadata,
                model details, and the threat assessment. Requires
                <code>pip install fpdf2</code>.</div>
        </div>
        <div class="faq-item">
            <div class="faq-q">Can I run VIGIL.AI on GPU?</div>
            <div class="faq-a">Yes. The FastAPI backend automatically uses CUDA if a compatible
                GPU and PyTorch CUDA build are present. GPU inference significantly reduces
                processing time for long surveillance feeds.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="support-card" style="margin-top:1.2rem">
    <div class="support-card-header">System Information</div>
    <div class="support-card-body">
        <div class="support-channel" style="padding:0.65rem 0;">
            <div style="font-family:'DM Mono',monospace;font-size:0.62rem;font-weight:500;letter-spacing:0.1em;color:#94a3b8;text-transform:uppercase;width:140px;flex-shrink:0;">Violence Model</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:0.88rem;font-weight:600;color:#1e293b;">R3D-18 &middot; 4-Class</div>
        </div>
        <div class="support-channel" style="padding:0.65rem 0;">
            <div style="font-family:'DM Mono',monospace;font-size:0.62rem;font-weight:500;letter-spacing:0.1em;color:#94a3b8;text-transform:uppercase;width:140px;flex-shrink:0;">Weapon Model</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:0.88rem;font-weight:600;color:#1e293b;">YOLOv8 &middot; 4-Class</div>
        </div>
        <div class="support-channel" style="padding:0.65rem 0;">
            <div style="font-family:'DM Mono',monospace;font-size:0.62rem;font-weight:500;letter-spacing:0.1em;color:#94a3b8;text-transform:uppercase;width:140px;flex-shrink:0;">Backend</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:0.88rem;font-weight:600;color:#1e293b;">FastAPI &middot; 127.0.0.1:8000</div>
        </div>
        <div class="support-channel" style="padding:0.65rem 0;border-bottom:none;">
            <div style="font-family:'DM Mono',monospace;font-size:0.62rem;font-weight:500;letter-spacing:0.1em;color:#94a3b8;text-transform:uppercase;width:140px;flex-shrink:0;">Temporal Smooth</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:0.88rem;font-weight:600;color:#1e293b;">5-Frame Majority Vote</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="margin-top:1.5rem;text-align:center;">
    <a href="#dashboard"
       style="display:inline-flex;align-items:center;gap:0.45rem;
              font-family:'DM Sans',sans-serif;font-size:0.85rem;font-weight:600;
              color:#fff;background:#2563eb;text-decoration:none;
              padding:0.65rem 1.5rem;border-radius:9px;
              box-shadow:0 3px 12px rgba(37,99,235,0.25);">
        &#8592; Back to Dashboard
    </a>
</div>
""", unsafe_allow_html=True)

md('</section>')
md('<div class="vigil-section-spacer"></div>')

# ═══════════════════════════════════════════════════════
#  HELP BAR + FOOTER  (all pages)
# ═══════════════════════════════════════════════════════
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

st.markdown("""
<div class="vigil-divider"></div>
<div class="vigil-footer">
    <div>VIGIL.AI &nbsp;&middot;&nbsp; Automated Violence &amp; Weapon Detection &nbsp;&middot;&nbsp; R3D-18 + YOLOv8</div>
    <div>FIGHT &nbsp;&middot;&nbsp; HOCKEY FIGHT &nbsp;&middot;&nbsp; MOVIE FIGHT &nbsp;&middot;&nbsp; NON-FIGHT &nbsp;&middot;&nbsp; WEAPONIZED</div>
</div>
""", unsafe_allow_html=True)