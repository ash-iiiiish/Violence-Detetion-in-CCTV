import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000/predict/"

st.set_page_config(
    page_title="VIGIL.AI — Automated Violence Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Syne:wght@700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0d1117 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 16px !important;
}

[data-testid="stAppViewContainer"] { background: #0d1117 !important; }
[data-testid="stHeader"]           { background: transparent !important; }
[data-testid="stToolbar"]          { display: none !important; }
#MainMenu, footer, header          { visibility: hidden !important; }
.stDeployButton                    { display: none !important; }

/* ── RESPONSIVE CONTAINER ── */
.block-container {
    padding: 2rem 4rem 5rem !important;
    max-width: 1500px !important;
}

@media (max-width: 1200px) {
    .block-container { padding: 1.8rem 2.5rem 4rem !important; }
}

@media (max-width: 768px) {
    .block-container { padding: 1.2rem 1rem 3rem !important; }
    html, body, [data-testid="stAppViewContainer"] { font-size: 14px !important; }
}

@media (max-width: 480px) {
    .block-container { padding: 0.8rem 0.6rem 2.5rem !important; }
    html, body, [data-testid="stAppViewContainer"] { font-size: 13px !important; }
}

::-webkit-scrollbar            { width: 4px; }
::-webkit-scrollbar-track      { background: #0d1117; }
::-webkit-scrollbar-thumb      { background: #2d3748; border-radius: 4px; }

/* ═══════════════════════════════
   NAV BAR — RESPONSIVE
═══════════════════════════════ */
.vigil-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.1rem 0 1.3rem;
    border-bottom: 1px solid #1e2a3a;
    margin-bottom: 2.5rem;
    flex-wrap: wrap;
    gap: 0.8rem;
}

.vigil-logo-area {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    flex-shrink: 0;
}

.vigil-eye-icon {
    width: 42px; height: 42px;
    flex-shrink: 0;
}

@media (max-width: 600px) {
    .vigil-eye-icon { width: 32px; height: 32px; }
}

.vigil-logo {
    font-family: 'Syne', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    color: #f8fafc;
    line-height: 1;
}

@media (max-width: 768px)  { .vigil-logo { font-size: 1.7rem; } }
@media (max-width: 480px)  { .vigil-logo { font-size: 1.4rem; } }

.vigil-logo-dot { color: #3b82f6; }

.vigil-nav-meta {
    display: flex;
    align-items: center;
    gap: 1.6rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #64748b;
    flex-wrap: wrap;
}

@media (max-width: 900px) {
    .vigil-nav-meta { gap: 1rem; font-size: 0.72rem; }
    .nav-sep        { display: none; }
}

@media (max-width: 600px) {
    .vigil-nav-meta { font-size: 0.65rem; gap: 0.6rem; }
}

.nav-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #94a3b8;
    font-weight: 600;
}

.nav-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 7px rgba(34,197,94,0.6);
    animation: navpulse 2.5s ease-in-out infinite;
    flex-shrink: 0;
}

@keyframes navpulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.35; }
}

.nav-sep { color: #2d3748; }

/* ═══════════════════════════════
   HERO — RESPONSIVE
═══════════════════════════════ */
.vigil-hero { margin-bottom: 2.2rem; }

.vigil-hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    color: #3b82f6;
    text-transform: uppercase;
    margin-bottom: 0.65rem;
}

.vigil-hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.6rem, 3.5vw, 2.9rem);
    font-weight: 800;
    color: #f1f5f9;
    line-height: 1.18;
    letter-spacing: -0.01em;
    margin-bottom: 0.75rem;
}

.vigil-hero-title span { color: #3b82f6; }

.vigil-hero-desc {
    font-size: clamp(0.88rem, 1.5vw, 1rem);
    color: #64748b;
    line-height: 1.7;
    max-width: 680px;
    font-weight: 400;
}

/* ═══════════════════════════════
   STATUS STRIP — RESPONSIVE
═══════════════════════════════ */
.status-strip {
    display: flex;
    background: #111827;
    border: 1px solid #1e2a3a;
    border-radius: 8px;
    margin-bottom: 2rem;
    overflow: hidden;
    flex-wrap: wrap;
}

.strip-item {
    flex: 1;
    min-width: 130px;
    padding: 0.9rem 1.3rem;
    border-right: 1px solid #1e2a3a;
    border-bottom: 0;
}

@media (max-width: 900px) {
    .strip-item { min-width: 45%; border-bottom: 1px solid #1e2a3a; }
    .strip-item:nth-child(2n) { border-right: none; }
    .strip-item:nth-last-child(-n+1) { border-bottom: none; }
}

@media (max-width: 480px) {
    .strip-item { min-width: 100%; border-right: none; }
    .strip-item:last-child { border-bottom: none; }
}

.strip-item:last-child { border-right: none; }

.strip-key {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.63rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    color: #4b5563;
    text-transform: uppercase;
    margin-bottom: 0.32rem;
}

.strip-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    color: #94a3b8;
}

.strip-val.online { color: #22c55e; }

/* ═══════════════════════════════
   CARDS
═══════════════════════════════ */
.card {
    background: #111827;
    border: 1px solid #1e2a3a;
    border-radius: 10px;
    padding: 1.5rem 1.7rem;
    position: relative;
}

@media (max-width: 600px) {
    .card { padding: 1.1rem 1.1rem; }
}

.card-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    color: #64748b;
    text-transform: uppercase;
    margin-bottom: 1.1rem;
    padding-bottom: 0.85rem;
    border-bottom: 1px solid #1e2a3a;
    display: flex;
    align-items: center;
    gap: 0.55rem;
}

.card-title::before {
    content: '';
    display: inline-block;
    width: 3px; height: 14px;
    background: #3b82f6;
    border-radius: 2px;
    flex-shrink: 0;
}

/* ═══════════════════════════════
   FILE UPLOADER
═══════════════════════════════ */
[data-testid="stFileUploader"] { background: transparent !important; }

[data-testid="stFileUploader"] > div {
    background: #0d1117 !important;
    border: 1.5px dashed #1e2a3a !important;
    border-radius: 8px !important;
    transition: all 0.25s ease !important;
    padding: 1.8rem 1.2rem !important;
}

[data-testid="stFileUploader"] > div:hover {
    border-color: #3b82f6 !important;
    background: rgba(59,130,246,0.03) !important;
}

[data-testid="stFileUploader"] label {
    color: #64748b !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}

[data-testid="stFileDropzone"] p,
[data-testid="stFileUploader"] section p {
    color: #64748b !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}

[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] .uploadedFile {
    color: #374151 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* ═══════════════════════════════
   BUTTON
═══════════════════════════════ */
.stButton > button {
    width: 100% !important;
    background: #1d4ed8 !important;
    color: #ffffff !important;
    border: 1px solid #2563eb !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.96rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    padding: 0.92rem 2rem !important;
    transition: all 0.2s ease !important;
    margin-top: 0.4rem !important;
}

.stButton > button:hover {
    background: #2563eb !important;
    border-color: #3b82f6 !important;
    box-shadow: 0 4px 22px rgba(59,130,246,0.3) !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active { transform: translateY(0) !important; }

/* ═══════════════════════════════
   VIDEO
═══════════════════════════════ */
[data-testid="stVideo"] video, video {
    border: 1px solid #1e2a3a !important;
    border-radius: 8px !important;
    width: 100% !important;
    background: #000 !important;
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
    background: #0d1117;
    border: 1px solid #1e2a3a;
    border-radius: 10px;
}

.globe-wrap {
    position: relative;
    width: 110px; height: 110px;
    flex-shrink: 0;
}

.globe-ring {
    position: absolute; inset: 0;
    border-radius: 50%;
    border: 1.5px solid rgba(59,130,246,0.2);
}

.globe-arc {
    position: absolute; inset: 0;
    border-radius: 50%;
    border: 2px solid transparent;
    border-top-color: #3b82f6;
    border-right-color: rgba(59,130,246,0.3);
    animation: globeSpin 1.4s linear infinite;
    box-shadow: 0 0 18px rgba(59,130,246,0.2);
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
    width: 18px; height: 18px;
    border-radius: 50%;
    background: radial-gradient(circle, #3b82f6, #1d4ed8 70%);
    box-shadow: 0 0 20px rgba(59,130,246,0.6), 0 0 40px rgba(59,130,246,0.2);
    animation: corePulse 1.4s ease-in-out infinite;
}

.globe-line {
    position: absolute; top: 50%; left: 50%;
    width: 45px; height: 1px;
    background: linear-gradient(90deg, rgba(59,130,246,0.6), transparent);
    transform-origin: left center;
    animation: radarSweep 2s linear infinite;
}
.globe-line:nth-child(5) { animation-delay: -0.66s; opacity: 0.5; }
.globe-line:nth-child(6) { animation-delay: -1.33s; opacity: 0.3; }

@keyframes globeSpin  { to { transform: rotate(360deg); } }
@keyframes globeSpinR { to { transform: rotate(-360deg); } }
@keyframes corePulse  {
    0%,100% { box-shadow: 0 0 20px rgba(59,130,246,0.6),0 0 40px rgba(59,130,246,0.2); }
    50%     { box-shadow: 0 0 30px rgba(59,130,246,0.9),0 0 60px rgba(59,130,246,0.35); }
}
@keyframes radarSweep { to { transform: rotate(360deg); } }

.globe-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 600;
    color: #3b82f6;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    text-align: center;
}

.globe-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #374151;
    letter-spacing: 0.1em;
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
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.73rem;
    color: #374151;
}

.globe-step-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #1e2a3a;
    flex-shrink: 0;
}

.globe-step.active .globe-step-dot {
    background: #3b82f6;
    box-shadow: 0 0 6px rgba(59,130,246,0.5);
}
.globe-step.active { color: #94a3b8; }
.globe-step.done .globe-step-dot { background: #22c55e; }
.globe-step.done  { color: #4b5563; }

/* ── ANALYSIS COMPLETE BADGE ── */
.analysis-done {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.75rem 1.2rem;
    background: rgba(34,197,94,0.07);
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 8px;
    margin-bottom: 0.5rem;
    animation: fadeUp 0.4s ease;
}

.analysis-done-icon {
    width: 22px; height: 22px;
    border-radius: 50%;
    background: rgba(34,197,94,0.15);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem;
    color: #22c55e;
    flex-shrink: 0;
}

.analysis-done-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    color: #22c55e;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.analysis-done-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #374151;
    margin-top: 0.1rem;
}

/* ═══════════════════════════════
   RESULT CARD — RESPONSIVE
═══════════════════════════════ */
.result-card {
    margin-top: 1.2rem;
    border: 1px solid #1e2a3a;
    border-radius: 10px;
    overflow: hidden;
    animation: fadeUp 0.4s ease;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

.result-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.95rem 1.5rem;
    background: #0f1923;
    border-bottom: 1px solid #1e2a3a;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.result-top-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    color: #4b5563;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.result-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.73rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    padding: 0.3rem 0.9rem;
    border-radius: 5px;
    text-transform: uppercase;
}

.badge-safe   { color:#22c55e; border:1px solid rgba(34,197,94,0.35);  background:rgba(34,197,94,0.08); }
.badge-fight  { color:#f59e0b; border:1px solid rgba(245,158,11,0.35); background:rgba(245,158,11,0.08); }
.badge-weapon {
    color:#ef4444; border:1px solid rgba(239,68,68,0.4); background:rgba(239,68,68,0.1);
    animation: threatpulse 1.2s ease-in-out infinite;
}

@keyframes threatpulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
    50%     { box-shadow: 0 0 0 4px rgba(239,68,68,0.15); }
}

.result-stats {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    background: #111827;
}

@media (max-width: 600px) {
    .result-stats { grid-template-columns: 1fr 1fr; }
    .r-stat:last-child { grid-column: span 2; border-right: none; border-top: 1px solid #1e2a3a; }
}

@media (max-width: 380px) {
    .result-stats { grid-template-columns: 1fr; }
    .r-stat { border-right: none !important; border-bottom: 1px solid #1e2a3a; }
    .r-stat:last-child { border-bottom: none; border-top: none; }
}

.r-stat { padding: 1.2rem 1.5rem; border-right: 1px solid #1e2a3a; }
.r-stat:last-child { border-right: none; }

.r-stat-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.67rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: #374151;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

.r-stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #e2e8f0;
    line-height: 1;
}

.r-stat-value.c-safe    { color: #22c55e; }
.r-stat-value.c-warning { color: #f59e0b; }
.r-stat-value.c-danger  { color: #ef4444; }

.r-stat-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #2d3748;
    margin-top: 0.3rem;
}

.conf-section {
    padding: 1rem 1.5rem 1.3rem;
    background: #0f1923;
    border-top: 1px solid #1e2a3a;
}

.conf-header {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #374151;
    margin-bottom: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.conf-header span:last-child { color: #94a3b8; font-size: 0.8rem; }

.conf-track { height: 5px; background: #1e2a3a; border-radius: 999px; overflow: hidden; }
.conf-fill  { height: 100%; border-radius: 999px; }

/* ═══════════════════════════════
   IDLE BOXES
═══════════════════════════════ */
.idle-box {
    background: #0d1117;
    border: 1.5px dashed #1e2a3a;
    border-radius: 8px;
    min-height: 160px;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 0.5rem; padding: 2rem; text-align: center;
}

.idle-icon  { font-size: 1.6rem; opacity: 0.2; margin-bottom: 0.2rem; }
.idle-title { font-family: 'Inter', sans-serif; font-size: 0.9rem; font-weight: 600; color: #2d3748; }
.idle-sub   { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #1f2937; letter-spacing: 0.08em; }

/* ═══════════════════════════════
   DETECTION TABLE — RESPONSIVE
═══════════════════════════════ */
.legend-table { width: 100%; border-collapse: collapse; margin-top: 0.3rem; }

.legend-table tr { border-bottom: 1px solid #161f2c; transition: background 0.15s; }
.legend-table tr:last-child { border-bottom: none; }
.legend-table tr:hover { background: rgba(255,255,255,0.02); }
.legend-table td { padding: 0.78rem 0.4rem; vertical-align: middle; }

.l-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 0.6rem;
    vertical-align: middle;
    flex-shrink: 0;
}

.l-class {
    font-family: 'Inter', sans-serif;
    font-weight: 700; color: #cbd5e1; font-size: 0.93rem;
}

.l-def {
    font-family: 'Inter', sans-serif;
    font-size: 0.83rem; font-weight: 500; color: #94a3b8;
    background: rgba(255,255,255,0.04);
    border: 1px solid #1e2a3a;
    border-radius: 5px;
    padding: 0.22rem 0.6rem;
    display: inline-block;
    margin-right: 0.4rem;
}

@media (max-width: 600px) {
    .l-def { font-size: 0.74rem; padding: 0.18rem 0.45rem; }
    .l-class { font-size: 0.82rem; }
}

.l-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; font-weight: 700;
    padding: 0.17rem 0.48rem;
    border-radius: 4px; letter-spacing: 0.08em;
    vertical-align: middle;
}

/* ═══════════════════════════════
   ERROR
═══════════════════════════════ */
.err-box {
    border: 1px solid rgba(239,68,68,0.3);
    background: rgba(239,68,68,0.06);
    border-radius: 8px;
    padding: 1.1rem 1.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem; color: #f87171; line-height: 1.7;
    margin-top: 1.2rem;
}

/* ═══════════════════════════════
   FOOTER — RESPONSIVE
═══════════════════════════════ */
.vigil-divider { height: 1px; background: #1e2a3a; margin: 2.5rem 0 1.2rem; }

.vigil-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; color: #1f2937; letter-spacing: 0.1em;
    padding-bottom: 0.5rem;
    flex-wrap: wrap; gap: 0.4rem;
}

@media (max-width: 600px) {
    .vigil-footer { flex-direction: column; align-items: flex-start; gap: 0.3rem; }
}

/* ═══════════════════════════════
   STREAMLIT COLUMN RESPONSIVE
═══════════════════════════════ */
@media (max-width: 768px) {
    [data-testid="column"] { min-width: 100% !important; }
    .stHorizontalBlock { flex-wrap: wrap !important; }
}
</style>
""", unsafe_allow_html=True)


# ── NAV ──────────────────────────────────────────────
st.markdown("""
<div class="vigil-nav">
    <div class="vigil-logo-area">
        <svg class="vigil-eye-icon" viewBox="0 0 44 44" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="22" cy="22" r="20" stroke="rgba(59,130,246,0.18)" stroke-width="1"/>
            <circle cx="22" cy="22" r="14" stroke="rgba(59,130,246,0.25)" stroke-width="1"/>
            <circle cx="22" cy="22" r="20" stroke="#3b82f6" stroke-width="1.5"
                    stroke-dasharray="28 97" stroke-linecap="round">
                <animateTransform attributeName="transform" type="rotate"
                    from="0 22 22" to="360 22 22" dur="3s" repeatCount="indefinite"/>
            </circle>
            <path d="M6 22 C10 13, 34 13, 38 22 C34 31, 10 31, 6 22Z"
                  fill="rgba(59,130,246,0.07)" stroke="#3b82f6" stroke-width="1.2"/>
            <circle cx="22" cy="22" r="6" fill="rgba(59,130,246,0.15)" stroke="#3b82f6" stroke-width="1.2"/>
            <circle cx="22" cy="22" r="2.5" fill="#3b82f6">
                <animate attributeName="r" values="2.5;3.2;2.5" dur="2s" repeatCount="indefinite"/>
            </circle>
            <line x1="2" y1="8" x2="8" y2="2" stroke="rgba(59,130,246,0.4)" stroke-width="1.2" stroke-linecap="round"/>
            <line x1="42" y1="8" x2="36" y2="2" stroke="rgba(59,130,246,0.4)" stroke-width="1.2" stroke-linecap="round"/>
            <line x1="2" y1="36" x2="8" y2="42" stroke="rgba(59,130,246,0.4)" stroke-width="1.2" stroke-linecap="round"/>
            <line x1="42" y1="36" x2="36" y2="42" stroke="rgba(59,130,246,0.4)" stroke-width="1.2" stroke-linecap="round"/>
        </svg>
        <span class="vigil-logo">VIGIL<span class="vigil-logo-dot">.</span>AI</span>
    </div>
    <div class="vigil-nav-meta">
        <div class="nav-status"><span class="nav-dot"></span>System Online</div>
        <span class="nav-sep">|</span>
        <div>R3D-18 &nbsp;·&nbsp; YOLOv8</div>
        <span class="nav-sep">|</span>
        <div>FastAPI Backend</div>
        <span class="nav-sep">|</span>
        <div>Real-Time Inference</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────
st.markdown("""
<div class="vigil-hero">
    <div class="vigil-hero-eyebrow">Automated Violence &amp; Weapon Detection System</div>
    <div class="vigil-hero-title">Intelligent <span>Threat Analysis</span><br>for Surveillance Footage</div>
    <div class="vigil-hero-desc">
        Upload a surveillance video to classify violent activity and detect weapons in real time.
        Powered by a 3D-CNN (R3D-18) for action recognition and YOLOv8 for weapon localisation,
        with frame-level annotation and confidence scoring.
    </div>
</div>
""", unsafe_allow_html=True)

# ── STATUS STRIP ─────────────────────────────────────
st.markdown("""
<div class="status-strip">
    <div class="strip-item">
        <div class="strip-key">Model Status</div>
        <div class="strip-val online">● Loaded &amp; Ready</div>
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
        label_visibility="hidden"
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
                <td style="width:36%"><span class="l-dot" style="background:#22c55e;box-shadow:0 0 6px rgba(34,197,94,0.5)"></span><span class="l-class">NonFight</span></td>
                <td>
                    <span class="l-def">No violent activity detected</span>
                    <span class="l-badge" style="color:#22c55e;border:1px solid rgba(34,197,94,0.35);background:rgba(34,197,94,0.1)">SAFE</span>
                </td>
            </tr>
            <tr>
                <td><span class="l-dot" style="background:#f59e0b;box-shadow:0 0 6px rgba(245,158,11,0.5)"></span><span class="l-class">Fight</span></td>
                <td>
                    <span class="l-def">Physical altercation between subjects</span>
                    <span class="l-badge" style="color:#f59e0b;border:1px solid rgba(245,158,11,0.35);background:rgba(245,158,11,0.1)">HIGH</span>
                </td>
            </tr>
            <tr>
                <td><span class="l-dot" style="background:#fb923c;box-shadow:0 0 6px rgba(251,146,60,0.5)"></span><span class="l-class">HockeyFight</span></td>
                <td>
                    <span class="l-def">Sport-context violent confrontation</span>
                    <span class="l-badge" style="color:#fb923c;border:1px solid rgba(251,146,60,0.35);background:rgba(251,146,60,0.1)">HIGH</span>
                </td>
            </tr>
            <tr>
                <td><span class="l-dot" style="background:#a78bfa;box-shadow:0 0 6px rgba(167,139,250,0.5)"></span><span class="l-class">MovieFight</span></td>
                <td>
                    <span class="l-def">Scripted / cinematic fight sequence</span>
                    <span class="l-badge" style="color:#a78bfa;border:1px solid rgba(167,139,250,0.35);background:rgba(167,139,250,0.1)">MED</span>
                </td>
            </tr>
            <tr>
                <td><span class="l-dot" style="background:#ef4444;box-shadow:0 0 6px rgba(239,68,68,0.5)"></span><span class="l-class">Weaponized</span></td>
                <td>
                    <span class="l-def">Knife · Handgun · Rifle · Launcher</span>
                    <span class="l-badge" style="color:#ef4444;border:1px solid rgba(239,68,68,0.35);background:rgba(239,68,68,0.12)">CRITICAL</span>
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
            <div class="idle-icon">⬡</div>
            <div class="idle-title">No Feed Uploaded</div>
            <div class="idle-sub">UPLOAD AN .MP4 FILE TO BEGIN</div>
        </div>
        """, unsafe_allow_html=True)
        analyze = False
    else:
        analyze = st.button("Run Threat Analysis", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None and analyze:

        # ── Show globe loader WHILE request runs ──────────
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

        # ── Run actual inference ───────────────────────────
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

        # ── Replace loader with "Analysis Complete" badge ──
        loader_slot.markdown("""
        <div class="analysis-done">
            <div class="analysis-done-icon">✓</div>
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
                bar_color  = "#ef4444"
                threat_lv  = "CRITICAL"
                threat_col = "#ef4444"
            elif "fight" in pred_lower:
                badge_cls  = "badge-fight"
                val_cls    = "c-warning"
                bar_color  = "#f59e0b"
                threat_lv  = "HIGH"
                threat_col = "#f59e0b"
            else:
                badge_cls  = "badge-safe"
                val_cls    = "c-safe"
                bar_color  = "#22c55e"
                threat_lv  = "NONE"
                threat_col = "#22c55e"

            st.markdown(f"""
            <div class="result-card">
                <div class="result-top">
                    <div class="result-top-label">Assessment Result</div>
                    <div class="result-badge {badge_cls}">{prediction}</div>
                </div>
                <div class="result-stats">
                    <div class="r-stat">
                        <div class="r-stat-label">Classification</div>
                        <div class="r-stat-value {val_cls}" style="font-size:1.08rem;margin-top:0.15rem">{prediction}</div>
                        <div class="r-stat-sub">R3D-18 Output</div>
                    </div>
                    <div class="r-stat">
                        <div class="r-stat-label">Confidence</div>
                        <div class="r-stat-value {val_cls}">{confidence}%</div>
                        <div class="r-stat-sub">Softmax Score</div>
                    </div>
                    <div class="r-stat">
                        <div class="r-stat-label">Threat Level</div>
                        <div class="r-stat-value" style="color:{threat_col}">{threat_lv}</div>
                        <div class="r-stat-sub">Risk Assessment</div>
                    </div>
                </div>
                <div class="conf-section">
                    <div class="conf-header">
                        <span>Confidence Score</span>
                        <span>{confidence}%</span>
                    </div>
                    <div class="conf-track">
                        <div class="conf-fill" style="width:{conf_int}%;background:linear-gradient(90deg,{bar_color}88,{bar_color})"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="card" style="margin-top:1.4rem"><div class="card-title">Output — Annotated Feed</div>', unsafe_allow_html=True)
            st.video(video_url)
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            err = error_msg if response is None else f"HTTP {response.status_code}"
            st.markdown(f"""
            <div class="err-box">
                ✕ &nbsp;Backend connection failed — {err}<br>
                Ensure the FastAPI server is running at 127.0.0.1:8000
            </div>
            """, unsafe_allow_html=True)

    elif uploaded_file is not None:
        st.markdown("""
        <div class="card" style="margin-top:1.4rem">
            <div class="card-title">Output — Annotated Feed</div>
            <div class="idle-box">
                <div class="idle-icon">▷</div>
                <div class="idle-title">Ready to Analyse</div>
                <div class="idle-sub">PRESS RUN THREAT ANALYSIS</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── FOOTER ───────────────────────────────────────────
st.markdown("""
<div class="vigil-divider"></div>
<div class="vigil-footer">
    <div>VIGIL.AI &nbsp;·&nbsp; Automated Violence &amp; Weapon Detection &nbsp;·&nbsp; R3D-18 + YOLOv8</div>
    <div>FIGHT &nbsp;·&nbsp; HOCKEY FIGHT &nbsp;·&nbsp; MOVIE FIGHT &nbsp;·&nbsp; NON-FIGHT &nbsp;·&nbsp; WEAPONIZED</div>
</div>
""", unsafe_allow_html=True)