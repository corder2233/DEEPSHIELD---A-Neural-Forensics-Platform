import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tempfile
import subprocess
import requests
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# Model Path
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mobilenet_deepfake.h5")

# ─────────────────────────────────────────────
# Background Image
# ─────────────────────────────────────────────
BG_IMAGE_URL = "https://images.unsplash.com/photo-1677442135703-1787eea5ce01?w=1920&q=80"

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DeepShield · AI Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Share+Tech+Mono&family=Rajdhani:wght@400;600&display=swap');

:root {{
    --neon-cyan:   #00f5ff;
    --neon-pink:   #ff00c8;
    --neon-red:    #ff1a1a;
    --neon-green:  #39ff14;
    --neon-amber:  #ffb400;
    --bg-void:     #010204;
    --bg-card:     rgba(0,8,18,0.82);
    --border-dim:  rgba(0,245,255,0.13);
    --border-glow: rgba(0,245,255,0.5);
    --text-primary:#e0f8ff;
    --text-muted:  #4a7a8a;
}}

*,*::before,*::after {{ box-sizing:border-box; }}
html,body,[class*="css"] {{ font-family:'Rajdhani',sans-serif; background-color:var(--bg-void); color:var(--text-primary); }}
.main {{ background-color:transparent !important; }}
section[data-testid="stSidebar"] {{ display:none !important; }}
.block-container {{ padding-top:0 !important; padding-left:1.5rem !important; padding-right:1.5rem !important; max-width:1400px !important; }}

/* BACKGROUNDS */
body::before {{
    content:''; position:fixed; inset:0; z-index:-3;
    background:radial-gradient(ellipse 80% 50% at 20% 20%,rgba(0,50,110,0.4) 0%,transparent 65%),
               radial-gradient(ellipse 60% 40% at 80% 80%,rgba(80,0,20,0.3) 0%,transparent 65%),#010204;
    pointer-events:none;
}}
.bg-hero-image {{
    position:fixed; top:0; left:0; width:100vw; height:100vh; z-index:-2;
    background-image:url('{BG_IMAGE_URL}'); background-size:cover; background-position:center;
    opacity:0.22; filter:hue-rotate(200deg) saturate(1.8) brightness(0.85); pointer-events:none;
}}
body::after {{
    content:''; position:fixed; inset:0; z-index:-1; opacity:0.12;
    background-image:linear-gradient(rgba(0,245,255,0.07) 1px,transparent 1px),linear-gradient(90deg,rgba(0,245,255,0.07) 1px,transparent 1px);
    background-size:48px 48px; animation:grid-drift 30s linear infinite; pointer-events:none;
}}
@keyframes grid-drift {{ 0% {{ background-position:0 0; }} 100% {{ background-position:48px 48px; }} }}
.scanlines {{ position:fixed; inset:0; z-index:0; pointer-events:none; background:repeating-linear-gradient(0deg,rgba(0,0,0,0) 0px,rgba(0,0,0,0) 3px,rgba(0,0,0,0.03) 4px); }}

/* HERO */
.hero-wrap {{ text-align:center; padding:3.5rem 1rem 1.8rem; position:relative; z-index:1; }}
.hero-eyebrow {{ font-family:'Share Tech Mono',monospace; font-size:0.65rem; letter-spacing:6px; color:var(--neon-cyan); text-transform:uppercase; opacity:0.85; margin-bottom:0.8rem; animation:fade-up 1.2s ease both; }}
@keyframes fade-up {{ from {{ opacity:0; transform:translateY(-6px); }} to {{ opacity:0.85; transform:translateY(0); }} }}
.hero-title {{ font-family:'Orbitron',sans-serif; font-size:clamp(3rem,8vw,5.5rem); font-weight:900; letter-spacing:8px; line-height:1; background:linear-gradient(135deg,#00f5ff 0%,#ffffff 45%,#ff1a1a 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; animation:hero-glow 4s ease-in-out infinite alternate; margin:0.3rem 0; }}
@keyframes hero-glow {{ from {{ filter:drop-shadow(0 0 18px rgba(0,245,255,0.35)); }} to {{ filter:drop-shadow(0 0 40px rgba(255,26,26,0.45)); }} }}
.hero-sub {{ font-family:'Share Tech Mono',monospace; font-size:0.78rem; letter-spacing:4px; color:var(--text-muted); text-transform:uppercase; margin-top:0.9rem; animation:fade-up 1.5s ease 0.2s both; }}
.hero-line {{ width:200px; height:1px; background:linear-gradient(90deg,transparent,var(--neon-cyan),var(--neon-red),transparent); margin:1.4rem auto 0; animation:line-pulse 3s ease-in-out infinite; }}
@keyframes line-pulse {{ 0%,100% {{ opacity:0.4; width:100px; }} 50% {{ opacity:1; width:260px; }} }}

/* CARDS */
.neon-card {{ background:var(--bg-card); border:1px solid var(--border-dim); border-radius:14px; padding:1.6rem 1.8rem; margin-bottom:1.2rem; position:relative; overflow:hidden; transition:border-color 0.25s,box-shadow 0.25s; }}
.neon-card::before {{ content:''; position:absolute; top:0; left:0; width:100%; height:2px; background:linear-gradient(90deg,transparent,var(--neon-cyan),transparent); opacity:0.55; }}
.neon-card:hover {{ border-color:var(--border-glow); box-shadow:0 0 22px rgba(0,245,255,0.09); }}
.neon-card-red::before {{ background:linear-gradient(90deg,transparent,var(--neon-red),transparent); }}
.neon-card-red:hover {{ border-color:rgba(255,26,26,0.45); box-shadow:0 0 22px rgba(255,26,26,0.09); }}
.neon-card-green::before {{ background:linear-gradient(90deg,transparent,var(--neon-green),transparent); }}

/* SECTION HEADERS */
.sec-header {{ font-family:'Orbitron',sans-serif; font-size:1rem; font-weight:700; letter-spacing:2px; color:var(--neon-cyan); text-transform:uppercase; margin-bottom:1rem; display:flex; align-items:center; gap:0.6rem; }}
.sec-header::after {{ content:''; flex:1; height:1px; background:linear-gradient(90deg,var(--border-glow),transparent); }}

/* BADGES */
.badge {{ font-family:'Share Tech Mono',monospace; font-size:0.58rem; letter-spacing:2.5px; background:rgba(0,245,255,0.08); border:1px solid rgba(0,245,255,0.25); border-radius:4px; padding:0.25rem 0.65rem; color:var(--neon-cyan); text-transform:uppercase; display:inline-block; margin-bottom:0.9rem; }}
.badge-red   {{ background:rgba(255,26,26,0.08);  border-color:rgba(255,26,26,0.25);  color:var(--neon-red);   }}
.badge-green {{ background:rgba(57,255,20,0.08);  border-color:rgba(57,255,20,0.25);  color:var(--neon-green); }}
.badge-amber {{ background:rgba(255,180,0,0.08);  border-color:rgba(255,180,0,0.25);  color:var(--neon-amber); }}

/* YOUTUBE PROXY BOX */
.yt-proxy-box {{ background:rgba(0,245,255,0.04); border:1px solid rgba(0,245,255,0.3); border-radius:12px; padding:1.2rem 1.5rem; margin-bottom:1.2rem; }}
.yt-proxy-box h4 {{ font-family:'Orbitron',sans-serif; font-size:0.78rem; font-weight:700; letter-spacing:2px; color:var(--neon-cyan); text-transform:uppercase; margin:0 0 0.6rem; }}
.yt-proxy-box p {{ font-family:'Rajdhani',sans-serif; font-size:0.92rem; color:#7aaabb; margin:0 0 0.5rem; line-height:1.55; }}
.yt-proxy-box code {{ font-family:'Share Tech Mono',monospace; font-size:0.78rem; background:rgba(0,245,255,0.1); border:1px solid rgba(0,245,255,0.2); border-radius:4px; padding:0.15rem 0.45rem; color:var(--neon-cyan); }}

/* PANEL HINT TEXT — clearly visible */
.panel-hint {{
    font-family:'Share Tech Mono',monospace;
    font-size:0.72rem;
    letter-spacing:1.5px;
    color:#5fd8e8;
    text-transform:uppercase;
    margin-top:0.85rem;
    padding:0.55rem 0.9rem;
    background:rgba(0,245,255,0.09);
    border:1px solid rgba(0,245,255,0.28);
    border-radius:7px;
    line-height:1.7;
}}
.panel-hint-red {{
    color:#e07070;
    background:rgba(255,26,26,0.08);
    border-color:rgba(255,26,26,0.28);
}}

/* RESULT PANELS */
.result-real {{ background:linear-gradient(135deg,rgba(0,28,14,0.94),rgba(0,48,24,0.88)); border:1px solid rgba(57,255,20,0.42); border-radius:16px; padding:2.5rem; text-align:center; box-shadow:0 0 45px rgba(57,255,20,0.14); animation:result-pop 0.4s ease both; }}
.result-fake {{ background:linear-gradient(135deg,rgba(28,0,0,0.94),rgba(55,0,0,0.88)); border:1px solid rgba(255,26,26,0.48); border-radius:16px; padding:2.5rem; text-align:center; box-shadow:0 0 45px rgba(255,26,26,0.14); animation:result-pop 0.4s ease both; }}
@keyframes result-pop {{ from {{ opacity:0; transform:translateY(8px); }} to {{ opacity:1; transform:translateY(0); }} }}
.verdict-text {{ font-family:'Orbitron',sans-serif; font-size:3rem; font-weight:900; letter-spacing:4px; text-shadow:0 0 28px currentColor; }}
.verdict-real {{ color:var(--neon-green); }}
.verdict-fake {{ color:var(--neon-red); }}
.conf-label   {{ font-family:'Share Tech Mono',monospace; font-size:0.65rem; letter-spacing:4px; color:var(--text-muted); text-transform:uppercase; margin-top:0.7rem; }}
.conf-value   {{ font-family:'Orbitron',sans-serif; font-size:2.5rem; font-weight:700; margin-top:0.2rem; }}
.conf-real    {{ color:var(--neon-green); text-shadow:0 0 18px rgba(57,255,20,0.65); }}
.conf-fake    {{ color:var(--neon-red);   text-shadow:0 0 18px rgba(255,26,26,0.65); }}

/* STAT GRID */
.stat-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:0.9rem; margin-top:1.2rem; }}
.stat-box  {{ background:rgba(0,245,255,0.04); border:1px solid var(--border-dim); border-radius:10px; padding:1rem; text-align:center; transition:border-color 0.2s; }}
.stat-box:hover {{ border-color:rgba(0,245,255,0.28); }}
.stat-box .s-val {{ font-family:'Orbitron',sans-serif; font-size:1.5rem; font-weight:700; color:var(--neon-cyan); display:block; }}
.stat-box .s-lbl {{ font-family:'Share Tech Mono',monospace; font-size:0.58rem; letter-spacing:2px; color:var(--text-muted); text-transform:uppercase; display:block; margin-top:0.3rem; }}

/* BUTTONS */
.stButton > button {{ background:linear-gradient(135deg,rgba(0,80,255,0.88),rgba(255,0,200,0.88)) !important; color:#ffffff !important; font-family:'Orbitron',sans-serif !important; font-size:0.72rem !important; letter-spacing:3px !important; border:1px solid rgba(0,180,255,0.5) !important; border-radius:8px !important; padding:0.75rem 1.5rem !important; width:100% !important; text-transform:uppercase !important; transition:box-shadow 0.2s,border-color 0.2s !important; box-shadow:0 0 18px rgba(0,80,255,0.3) !important; }}
.stButton > button:hover {{ background:linear-gradient(135deg,rgba(0,120,255,0.98),rgba(255,0,220,0.98)) !important; box-shadow:0 0 32px rgba(255,0,200,0.5) !important; border-color:var(--neon-pink) !important; }}

/* TEXT INPUT */
.stTextInput > div > div > input {{ background:rgba(0,16,32,0.9) !important; border:1px solid var(--border-dim) !important; border-radius:8px !important; color:var(--neon-cyan) !important; font-family:'Share Tech Mono',monospace !important; font-size:0.8rem !important; }}
.stTextInput > div > div > input:focus {{ border-color:var(--neon-cyan) !important; box-shadow:0 0 12px rgba(0,245,255,0.25) !important; }}

/* STICKY TABS */
div[data-baseweb="tab-list"] {{ background:rgba(1,2,4,0.96) !important; border-bottom:1px solid rgba(0,245,255,0.13) !important; border-radius:0 !important; gap:0 !important; padding:0 1rem !important; position:sticky; top:0; z-index:999; box-shadow:0 4px 24px rgba(0,0,0,0.65); }}
div[data-baseweb="tab"] {{ font-family:'Orbitron',sans-serif !important; font-size:0.67rem !important; letter-spacing:2.5px !important; color:var(--text-muted) !important; border-radius:0 !important; padding:1rem 1.5rem !important; border-bottom:2px solid transparent !important; transition:color 0.2s,background 0.2s !important; text-transform:uppercase !important; }}
div[data-baseweb="tab"]:hover {{ color:var(--neon-cyan) !important; background:rgba(0,245,255,0.04) !important; }}
div[aria-selected="true"] {{ background:transparent !important; color:var(--neon-cyan) !important; border-bottom:2px solid var(--neon-cyan) !important; box-shadow:none !important; }}
div[data-testid="stTabPanel"] {{ padding-top:2rem !important; }}

/* MISC */
.stSlider > div {{ filter:hue-rotate(160deg) saturate(2); }}
.stExpander {{ background:rgba(0,245,255,0.03) !important; border:1px solid var(--border-dim) !important; border-radius:10px !important; }}
div[data-testid="stFileUploader"] {{ background:rgba(0,16,32,0.55) !important; border:1px dashed rgba(0,245,255,0.28) !important; border-radius:10px !important; }}
div[data-testid="stFileUploader"]:hover {{ border-color:rgba(0,245,255,0.55) !important; }}

/* GUIDE STEPS */
.guide-step {{ display:flex; gap:1.2rem; align-items:flex-start; margin-bottom:1.1rem; padding:1rem 1.4rem; background:rgba(0,245,255,0.025); border:1px solid var(--border-dim); border-radius:10px; transition:border-color 0.2s; }}
.guide-step:hover {{ border-color:rgba(0,245,255,0.32); }}
.step-num {{ font-family:'Orbitron',sans-serif; font-size:1.4rem; font-weight:900; color:var(--neon-cyan); opacity:0.38; line-height:1; min-width:30px; }}
.step-body h4 {{ font-family:'Orbitron',sans-serif; font-size:0.76rem; font-weight:700; letter-spacing:2px; color:var(--neon-cyan); text-transform:uppercase; margin:0 0 0.3rem; }}
.step-body p  {{ font-family:'Rajdhani',sans-serif; font-size:0.92rem; color:#7aaabb; margin:0; line-height:1.55; }}

/* AWARENESS CARDS */
.aware-card {{ background:rgba(255,26,26,0.04); border:1px solid rgba(255,26,26,0.14); border-radius:10px; padding:1.2rem 1.4rem; margin-bottom:0.95rem; transition:border-color 0.2s; }}
.aware-card:hover {{ border-color:rgba(255,26,26,0.38); }}
.aware-card h4 {{ font-family:'Orbitron',sans-serif; font-size:0.75rem; font-weight:700; letter-spacing:2px; color:var(--neon-red); text-transform:uppercase; margin:0 0 0.45rem; }}
.aware-card p  {{ font-family:'Rajdhani',sans-serif; font-size:0.92rem; color:#bb8888; margin:0; line-height:1.58; }}

/* HELPLINE CARDS */
.helpline-card {{ background:rgba(255,180,0,0.04); border:1px solid rgba(255,180,0,0.18); border-radius:10px; padding:1.1rem 1.4rem; margin-bottom:0.85rem; transition:border-color 0.2s; display:flex; gap:1rem; align-items:center; }}
.helpline-card:hover {{ border-color:rgba(255,180,0,0.45); }}
.helpline-icon {{ font-size:1.7rem; min-width:40px; text-align:center; }}
.helpline-info h4 {{ font-family:'Orbitron',sans-serif; font-size:0.7rem; font-weight:700; letter-spacing:2px; color:var(--neon-amber); text-transform:uppercase; margin:0 0 0.25rem; }}
.helpline-info p  {{ font-family:'Share Tech Mono',monospace; font-size:0.78rem; color:#c0a070; margin:0; }}
.helpline-info a  {{ color:var(--neon-cyan); text-decoration:none; font-size:0.7rem; letter-spacing:1px; }}
.helpline-info a:hover {{ color:#fff; text-decoration:underline; }}

/* PROCESS TIMELINE */
.process-step {{ display:flex; gap:1rem; margin-bottom:0; position:relative; }}
.process-step:not(:last-child)::after {{ content:''; position:absolute; left:19px; top:40px; width:2px; height:calc(100% - 8px); background:linear-gradient(180deg,rgba(0,245,255,0.35),rgba(0,245,255,0.04)); }}
.process-circle {{ width:38px; height:38px; border-radius:50%; background:rgba(0,245,255,0.07); border:1px solid rgba(0,245,255,0.38); display:flex; align-items:center; justify-content:center; font-family:'Orbitron',sans-serif; font-size:0.68rem; font-weight:700; color:var(--neon-cyan); flex-shrink:0; margin-top:0.2rem; }}
.process-content {{ padding:0 0 1.4rem; flex:1; }}
.process-content h4 {{ font-family:'Orbitron',sans-serif; font-size:0.7rem; font-weight:700; letter-spacing:1.5px; color:var(--neon-cyan); text-transform:uppercase; margin:0 0 0.25rem; }}
.process-content p  {{ font-family:'Rajdhani',sans-serif; font-size:0.87rem; color:#7aaabb; margin:0; line-height:1.52; }}

/* VIDEO LABEL */
.video-label {{ font-family:'Orbitron',sans-serif; font-size:0.68rem; letter-spacing:3px; color:var(--neon-cyan); text-transform:uppercase; margin-bottom:0.5rem; }}

/* ★ WATERMARK — BOTTOM LEFT */
.watermark {{
    position:fixed;
    bottom:14px;
    left:18px;
    right:auto;
    z-index:9999;
    font-family:'Share Tech Mono',monospace;
    font-size:0.62rem;
    letter-spacing:1.5px;
    color:rgba(0,245,255,0.7);
    text-align:left;
    line-height:1.65;
    pointer-events:none;
    border-right:2px solid rgba(0,245,255,0.55);
    border-left:none;
    padding-right:9px;
    padding-left:0;
    text-transform:uppercase;
    text-shadow:0 0 10px rgba(0,245,255,0.45);
}}

/* FOOTER */
.footer-bar {{ display:flex; justify-content:space-between; flex-wrap:wrap; gap:0.5rem; font-family:'Share Tech Mono',monospace; font-size:0.6rem; color:rgba(0,245,255,0.2); border-top:1px solid rgba(0,245,255,0.07); padding-top:1rem; margin-top:3rem; }}

/* RESPONSIVE */
@media (max-width:768px) {{
    .hero-title {{ font-size:2.4rem; letter-spacing:4px; }}
    .stat-grid {{ grid-template-columns:1fr; }}
    .block-container {{ padding-left:0.7rem !important; padding-right:0.7rem !important; }}
    .helpline-card {{ flex-direction:column; }}
    div[data-baseweb="tab"] {{ padding:0.75rem !important; font-size:0.55rem !important; }}
    .watermark {{ font-size:0.52rem; }}
}}
@media (max-width:480px) {{
    .hero-title {{ font-size:1.7rem; letter-spacing:2px; }}
    .verdict-text {{ font-size:1.8rem; }}
    .conf-value {{ font-size:1.7rem; }}
}}
</style>
""", unsafe_allow_html=True)

# Overlays
st.markdown('<div class="bg-hero-image"></div>', unsafe_allow_html=True)
st.markdown('<div class="scanlines"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="watermark">
    Developed &amp; Designed by<br>
    SUBASINI &amp; SHAKTI
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
IMG_SIZE    = 128
FRAME_COUNT = 10

# ─────────────────────────────────────────────
# Model Loading  (UNCHANGED)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"❌ mobilenet_deepfake.h5 not found at: {MODEL_PATH}"
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                      loss="binary_crossentropy", metrics=["accuracy"])
        return model, None
    except Exception:
        pass
    try:
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.regularizers import l2
        base  = MobileNetV2(weights=None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
        model = Sequential([
            base, GlobalAveragePooling2D(), BatchNormalization(), Dropout(0.3),
            Dense(256, activation="relu", kernel_regularizer=l2(0.01)),
            Dropout(0.3), Dense(1, activation="sigmoid", dtype=tf.float32)
        ])
        model.load_weights(MODEL_PATH)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                      loss="binary_crossentropy", metrics=["accuracy"])
        return model, None
    except Exception as e:
        return None, f"Model load failed.\nResave with: model.save('mobilenet_deepfake.keras')\n\nError: {e}"

# ─────────────────────────────────────────────
# Frame Extraction  (UNCHANGED)
# ─────────────────────────────────────────────
def extract_frames(video_path, n=FRAME_COUNT):
    cap    = cv2.VideoCapture(video_path)
    frames = []
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step   = max(total // n, 1)
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = preprocess_input(frame.astype(np.float32))
        frames.append(frame)
    cap.release()
    return np.array(frames)

# ─────────────────────────────────────────────
# Prediction  (UNCHANGED)
# ─────────────────────────────────────────────
def predict_video(video_path, model, threshold=0.5, n_frames=FRAME_COUNT):
    frames = extract_frames(video_path, n_frames)
    if len(frames) == 0:
        return None, 0, []
    probs      = model.predict(frames, verbose=0).flatten()
    avg_prob   = float(np.mean(probs))
    label      = "Fake" if avg_prob > threshold else "Real"
    confidence = avg_prob if avg_prob > threshold else 1 - avg_prob
    return label, round(confidence * 100, 2), probs.tolist()

# ─────────────────────────────────────────────
# ★ YouTube Download — Cloud-Compatible via Proxy
#
# WHY IT FAILS ON CLOUD:
#   YouTube blocks all datacenter IP ranges (AWS, GCP, Azure, Hetzner, etc.)
#   from downloading content.  Local machines work because home/office IPs
#   are not in those blocked ranges.
#
# SOLUTION — 3-TIER PROXY STRATEGY:
#   Tier 1: Piped API   — open-source YT frontend, returns direct CDN URLs
#   Tier 2: Invidious API — another open-source YT proxy, same principle
#   Tier 3: yt-dlp with --impersonate chrome (last resort)
#
# For direct .mp4 URLs: always works via requests.
# ─────────────────────────────────────────────

DIRECT_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")

PIPED_INSTANCES = [
    "https://pipedapi.kavin.rocks",
    "https://pipedapi.tokhmi.xyz",
    "https://piped-api.garudalinux.org",
    "https://api.piped.projectsegfau.lt",
]
INVIDIOUS_INSTANCES = [
    "https://invidious.snopyta.org",
    "https://invidious.kavin.rocks",
    "https://vid.puffyan.us",
    "https://invidious.nerdvpn.de",
]
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
_HDR = {"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9"}


def _extract_yt_id(url: str):
    import re
    m = re.search(r"(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None


def _stream_save(stream_url: str, save_path: str) -> str:
    with requests.get(stream_url, headers=_HDR, stream=True, timeout=90) as r:
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=2 << 20):
                f.write(chunk)
    if os.path.getsize(save_path) < 50_000:
        raise ValueError("Downloaded file too small — stream likely failed")
    return save_path


def _piped(vid_id: str, save_path: str) -> str:
    last = ""
    for inst in PIPED_INSTANCES:
        try:
            r = requests.get(f"{inst}/streams/{vid_id}", headers=_HDR, timeout=15)
            r.raise_for_status()
            data    = r.json()
            streams = data.get("videoStreams", [])
            chosen  = None
            for s in streams:
                if "mp4" in s.get("format", "").lower():
                    chosen = s
                    if any(q in s.get("quality", "") for q in ["720", "480", "360"]):
                        break
            if chosen is None and streams:
                chosen = streams[0]
            if chosen is None:
                raise ValueError("No streams in Piped response")
            return _stream_save(chosen["url"], save_path)
        except Exception as e:
            last = str(e)
    raise RuntimeError(f"Piped failed: {last}")


def _invidious(vid_id: str, save_path: str) -> str:
    last = ""
    for inst in INVIDIOUS_INSTANCES:
        try:
            r = requests.get(f"{inst}/api/v1/videos/{vid_id}", headers=_HDR, timeout=15)
            r.raise_for_status()
            data    = r.json()
            fmts    = data.get("formatStreams", []) + data.get("adaptiveFormats", [])
            chosen  = None
            for f in fmts:
                if f.get("container") == "mp4":
                    chosen = f
                    if any(q in f.get("qualityLabel", "") for q in ["720p", "480p", "360p"]):
                        break
            if chosen is None and fmts:
                chosen = fmts[0]
            if chosen is None:
                raise ValueError("No streams in Invidious response")
            url = chosen.get("url") or chosen.get("adaptiveUrl", "")
            if not url:
                raise ValueError("Empty URL from Invidious")
            return _stream_save(url, save_path)
        except Exception as e:
            last = str(e)
    raise RuntimeError(f"Invidious failed: {last}")


def _ytdlp(url: str, save_path: str) -> str:
    base = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
        "--merge-output-format", "mp4",
        "-o", save_path,
        "--quiet", "--no-warnings",
        "--socket-timeout", "30",
        "--retries", "2",
    ]
    err = ""
    for cmd in [base + ["--impersonate", "chrome", url], base + [url]]:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        err = res.stderr or res.stdout
        if res.returncode == 0:
            if os.path.exists(save_path) and os.path.getsize(save_path) >= 50_000:
                return save_path
            d    = os.path.dirname(save_path)
            mp4s = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".mp4")]
            if mp4s:
                biggest = max(mp4s, key=os.path.getsize)
                if os.path.getsize(biggest) >= 50_000:
                    return biggest
    raise RuntimeError(err or "yt-dlp produced no output file")


def download_video(url: str, save_path: str) -> str:
    """Route video download through the correct method. Returns saved file path."""
    url = url.strip()

    # Case 1: Direct .mp4 / video file URL
    if any(url.split("?")[0].lower().endswith(ext) for ext in DIRECT_EXTS):
        try:
            return _stream_save(url, save_path)
        except Exception as e:
            raise RuntimeError(f"Direct URL download failed: {e}")

    # Case 2: YouTube → 3-tier proxy
    if any(d in url for d in ("youtube.com", "youtu.be")):
        vid_id = _extract_yt_id(url)
        if not vid_id:
            raise RuntimeError(
                "Could not parse a YouTube video ID from the URL.\n"
                "Use a standard link: `https://www.youtube.com/watch?v=VIDEO_ID`"
            )
        errors = []
        for name, fn, arg in [("Piped", _piped, vid_id),
                               ("Invidious", _invidious, vid_id),
                               ("yt-dlp", _ytdlp, url)]:
            try:
                return fn(arg, save_path)
            except Exception as e:
                errors.append(f"{name}: {e}")

        raise RuntimeError(
            "**All YouTube proxy methods failed.**\n\n"
            "YouTube blocks cloud-server IPs from downloading. The three automatic "
            "bypass methods (Piped, Invidious, yt-dlp) were all attempted.\n\n"
            "**✅ Easy workarounds:**\n"
            "1. **Download & upload:** Use [yt-dlp](https://github.com/yt-dlp/yt-dlp) "
            "or a browser extension to save the video as `.mp4`, then upload it with the "
            "**Upload File** panel — this always works.\n"
            "2. **Direct link:** Share the video to Google Drive/Dropbox and paste the "
            "public `.mp4` download link here.\n"
            "3. **Run locally:** On your own machine, YouTube downloads always succeed.\n\n"
            f"*Details: {' | '.join(errors[:2])}*"
        )

    # Case 3: Other platform (Vimeo, Dailymotion, etc.) → yt-dlp
    try:
        return _ytdlp(url, save_path)
    except Exception as e:
        raise RuntimeError(
            f"Download failed for this URL: {e}\n\n"
            "💡 Download the video manually and use the **Upload File** panel."
        )


# ─────────────────────────────────────────────
# Plotly helpers
# ─────────────────────────────────────────────
_PBG  = "rgba(0,0,0,0)"
_PFNT = dict(family="Share Tech Mono, monospace", color="#4a7a8a", size=11)
_PGRD = "rgba(0,245,255,0.06)"
_PAXC = "#4a7a8a"
_PTIT = dict(family="Orbitron", size=11, color="#4a7a8a")
_CFG  = {"displayModeBar": False}

def _base_layout(**kw):
    return dict(paper_bgcolor=_PBG, plot_bgcolor="rgba(0,18,36,0.3)",
                margin=dict(t=28,b=30,l=42,r=18), height=220, font=_PFNT,
                showlegend=False, **kw)

def plotly_gauge(confidence, label):
    color = "#39ff14" if label == "Real" else "#ff1a1a"
    fig   = go.Figure(go.Indicator(
        mode="gauge+number", value=confidence,
        number={"suffix":"%","font":{"family":"Orbitron, sans-serif","size":28,"color":color}},
        gauge={
            "axis":{"range":[0,100],"tickfont":{"color":_PAXC,"size":10},"tickwidth":1,"tickcolor":_PAXC},
            "bar":{"color":color,"thickness":0.22},"bgcolor":"rgba(0,0,0,0)","borderwidth":0,
            "steps":[{"range":[0,40],"color":"rgba(57,255,20,0.05)"},
                     {"range":[40,70],"color":"rgba(255,180,0,0.05)"},
                     {"range":[70,100],"color":"rgba(255,26,26,0.07)"}],
            "threshold":{"line":{"color":color,"width":3},"thickness":0.75,"value":confidence},
        },
        title={"text":f"CONFIDENCE · {label.upper()}","font":{**_PTIT}},
    ))
    fig.update_layout(paper_bgcolor=_PBG,plot_bgcolor=_PBG,margin=dict(t=28,b=8,l=8,r=8),height=220,font=_PFNT)
    return fig

def plotly_frame_line(frame_probs, threshold):
    xs     = list(range(1, len(frame_probs)+1))
    colors = ["#ff1a1a" if p > threshold else "#39ff14" for p in frame_probs]
    fig    = go.Figure()
    fig.add_trace(go.Scatter(x=xs,y=frame_probs,mode="lines+markers",
        line=dict(color="#00f5ff",width=2),
        marker=dict(color=colors,size=8,line=dict(color="#000",width=1)),
        fill="tozeroy",fillcolor="rgba(0,245,255,0.04)"))
    fig.add_hline(y=threshold,line_dash="dot",line_color="#ffb400",
                  annotation_text=f"Threshold {threshold:.2f}",
                  annotation_font=dict(color="#ffb400",size=10))
    fig.update_layout(**_base_layout(
        xaxis=dict(title="Frame",gridcolor=_PGRD,color=_PAXC),
        yaxis=dict(title="P(Fake)",range=[0,1],gridcolor=_PGRD,color=_PAXC),
        title=dict(text="FRAME-LEVEL FAKE PROBABILITY",font=_PTIT,x=0.5)))
    return fig

def plotly_histogram(frame_probs):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=frame_probs,nbinsx=8,
        marker=dict(color=frame_probs,colorscale=[[0,"#39ff14"],[0.5,"#ffb400"],[1,"#ff1a1a"]],
                    line=dict(color="#000",width=1))))
    fig.update_layout(**_base_layout(
        xaxis=dict(title="Fake Prob",gridcolor=_PGRD,color=_PAXC),
        yaxis=dict(title="Frames",gridcolor=_PGRD,color=_PAXC),
        title=dict(text="SCORE DISTRIBUTION",font=_PTIT,x=0.5)))
    return fig

def plotly_donut(confidence, label):
    real_p = confidence if label == "Real" else 100-confidence
    fig    = go.Figure(go.Pie(values=[real_p,100-real_p],labels=["Real","Fake"],hole=0.68,
        marker=dict(colors=["#39ff14","#ff1a1a"],line=dict(color=["#000"],width=2)),
        textinfo="none",hoverinfo="label+percent"))
    fig.update_layout(paper_bgcolor=_PBG,plot_bgcolor=_PBG,margin=dict(t=28,b=8,l=8,r=8),
        height=220,font=_PFNT,showlegend=True,
        legend=dict(font=dict(color=_PAXC,size=10),bgcolor="rgba(0,0,0,0)"),
        title=dict(text="REAL vs FAKE SPLIT",font=_PTIT,x=0.5))
    return fig

def plotly_risk(frame_probs):
    xs  = list(range(1, len(frame_probs)+1))
    fig = go.Figure()
    fig.add_hrect(y0=0,  y1=0.4,fillcolor="rgba(57,255,20,0.05)", line_width=0)
    fig.add_hrect(y0=0.4,y1=0.6,fillcolor="rgba(255,180,0,0.06)", line_width=0)
    fig.add_hrect(y0=0.6,y1=1,  fillcolor="rgba(255,26,26,0.06)",  line_width=0)
    fig.add_trace(go.Bar(x=xs,y=frame_probs,
        marker=dict(color=frame_probs,
                    colorscale=[[0,"#39ff14"],[0.5,"#ffb400"],[1,"#ff1a1a"]],
                    line=dict(color="rgba(0,0,0,0.4)",width=0.5))))
    fig.update_layout(**_base_layout(
        xaxis=dict(title="Frame",gridcolor=_PGRD,color=_PAXC),
        yaxis=dict(title="Risk",range=[0,1],gridcolor=_PGRD,color=_PAXC),
        title=dict(text="PER-FRAME RISK",font=_PTIT,x=0.5)))
    return fig

# ─────────────────────────────────────────────
# Result renderer
# ─────────────────────────────────────────────
def show_result(label, confidence, frame_probs, threshold):
    if label == "Real":
        st.markdown(f"""
        <div class="result-real">
            <div class="verdict-text verdict-real">◈ AUTHENTIC</div>
            <div class="conf-label">Confidence Score</div>
            <div class="conf-value conf-real">{confidence}%</div>
            <div style="margin-top:0.8rem;font-family:'Share Tech Mono',monospace;font-size:0.62rem;
                        letter-spacing:3px;color:rgba(57,255,20,0.5);">NO SYNTHETIC SIGNATURES DETECTED</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-fake">
            <div class="verdict-text verdict-fake">⚠ DEEPFAKE</div>
            <div class="conf-label">Confidence Score</div>
            <div class="conf-value conf-fake">{confidence}%</div>
            <div style="margin-top:0.8rem;font-family:'Share Tech Mono',monospace;font-size:0.62rem;
                        letter-spacing:3px;color:rgba(255,26,26,0.5);">SYNTHETIC MANIPULATION DETECTED</div>
        </div>""", unsafe_allow_html=True)

    fake_frames = sum(1 for p in frame_probs if p > threshold)
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-box"><span class="s-val">{fake_frames}/{len(frame_probs)}</span><span class="s-lbl">Flagged Frames</span></div>
        <div class="stat-box"><span class="s-val">{float(np.mean(frame_probs)):.2f}</span><span class="s-lbl">Avg Fake Score</span></div>
        <div class="stat-box"><span class="s-val">{float(np.std(frame_probs)):.2f}</span><span class="s-lbl">Score Volatility</span></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-header">📊 Analysis Graphs</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plotly_gauge(confidence, label),           use_container_width=True, config=_CFG)
        st.plotly_chart(plotly_histogram(frame_probs),             use_container_width=True, config=_CFG)
    with c2:
        st.plotly_chart(plotly_frame_line(frame_probs, threshold), use_container_width=True, config=_CFG)
        st.plotly_chart(plotly_risk(frame_probs),                  use_container_width=True, config=_CFG)
    c3, _ = st.columns([1,1])
    with c3:
        st.plotly_chart(plotly_donut(confidence, label), use_container_width=True, config=_CFG)

# ─────────────────────────────────────────────
# SVG Illustration
# ─────────────────────────────────────────────
def hero_illustration():
    return """
    <div style="text-align:center;padding:0.4rem 0 0.6rem;">
    <svg viewBox="0 0 520 140" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:500px;opacity:0.44;">
      <defs>
        <filter id="gF"><feGaussianBlur stdDeviation="2.2" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
      </defs>
      <g stroke="#00f5ff" stroke-width="0.6" opacity="0.32" fill="none">
        <polyline points="0,70 60,70 80,50 140,50 160,70 220,70"/>
        <polyline points="220,70 240,50 300,50 320,70 380,70 400,50 460,50 480,70 520,70"/>
        <polyline points="80,50 80,22 120,22 140,6 200,6"/>
        <polyline points="300,50 300,22 340,22 360,6 420,6 440,22 480,22"/>
        <polyline points="160,70 160,100 200,100 220,120 280,120"/>
        <polyline points="360,70 360,100 400,100 420,120 480,120"/>
      </g>
      <g fill="#00f5ff" filter="url(#gF)">
        <circle cx="80"  cy="50" r="3.2" opacity="0.85"/>
        <circle cx="160" cy="70" r="3.2" opacity="0.85"/>
        <circle cx="240" cy="50" r="3.2" opacity="0.85"/>
        <circle cx="320" cy="70" r="3.2" opacity="0.85"/>
        <circle cx="400" cy="50" r="3.2" opacity="0.85"/>
        <circle cx="480" cy="70" r="3.2" opacity="0.85"/>
      </g>
      <ellipse cx="260" cy="70" rx="46" ry="28" stroke="#ff1a1a" stroke-width="1.1" fill="none" opacity="0.48" filter="url(#gF)"/>
      <ellipse cx="260" cy="70" rx="26" ry="16" stroke="#00f5ff" stroke-width="0.8" fill="none" opacity="0.55" filter="url(#gF)"/>
      <circle  cx="260" cy="70" r="6.5" fill="#ff1a1a" opacity="0.65" filter="url(#gF)"/>
      <line x1="214" y1="70" x2="248" y2="70" stroke="#ff1a1a" stroke-width="0.8" opacity="0.45"/>
      <line x1="272" y1="70" x2="306" y2="70" stroke="#ff1a1a" stroke-width="0.8" opacity="0.45"/>
    </svg></div>"""

# ════════════════════════════════════════════
#  HERO
# ════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">⬡ Neural Forensics Platform · v2.0</div>
    <div class="hero-title">DEEPSHIELD</div>
    <div class="hero-sub">Frame-level · AI Forensics · Synthetic Media Detection</div>
    <div class="hero-line"></div>
</div>""", unsafe_allow_html=True)
st.markdown(hero_illustration(), unsafe_allow_html=True)

# ════════════════════════════════════════════
#  MODEL LOAD
# ════════════════════════════════════════════
with st.spinner("⟳ Initialising AI model…"):
    model, load_error = load_model()
if model is not None:
    st.success("✅  AI Forensics Engine · Ready")
else:
    st.error(load_error)
    st.stop()

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════
#  SETTINGS
# ════════════════════════════════════════════
with st.expander("⚙️  Detection Settings", expanded=False):
    sc1, sc2 = st.columns(2)
    with sc1:
        frame_count = st.slider("Frames to sample", 5, 30, FRAME_COUNT,
                                help="More frames = higher accuracy, slower speed")
    with sc2:
        threshold = st.slider("Detection threshold", 0.30, 0.70, 0.50, 0.01,
                              help="Lower = more sensitive to fakes")

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════
tab_analyze, tab_guide, tab_awareness = st.tabs([
    "___ 🔬  Analyze Video___",
    "___ __📖  User Guide__ ___",
    "___ 🛡️  Awareness & Help___",
])

# ──────────────────────────────────────────
#  TAB 1: ANALYZE
# ──────────────────────────────────────────
with tab_analyze:

    # YouTube proxy explanation banner
    st.markdown("""
    <div class="yt-proxy-box">
        <h4>🔗 YouTube Links Supported via Automatic Proxy</h4>
        <p>
            Cloud servers are normally blocked by YouTube (HTTP 403 Forbidden).
            DeepShield automatically bypasses this using
            <strong style="color:#00f5ff;">Piped</strong> and
            <strong style="color:#00f5ff;">Invidious</strong> — open-source YouTube
            proxy APIs that return a direct CDN stream the cloud server
            <em>can</em> fetch. Just paste any standard YouTube link and click Scan URL.
        </p>
        <p style="font-size:0.83rem;color:#5a8a9a;">
            ⚡ <strong style="color:#ffb400;">If proxy fails</strong> — download the video
            with <a href="https://github.com/yt-dlp/yt-dlp" target="_blank"
            style="color:#00f5ff;">yt-dlp</a> or a browser extension and use the
            <strong style="color:#ff6666;">Upload File</strong> panel (always 100% reliable).
            Google Drive / Dropbox <code>.mp4</code> links also work directly.
        </p>
    </div>
    """, unsafe_allow_html=True)

    left_col, right_col = st.columns(2, gap="large")

    # LEFT — URL panel
    with left_col:
        st.markdown('<div class="neon-card">', unsafe_allow_html=True)
        st.markdown('<span class="badge">🔗 Stream URL</span>', unsafe_allow_html=True)
        st.markdown('<div class="sec-header" style="font-size:0.85rem;">Analyze via Link</div>',
                    unsafe_allow_html=True)
        url = st.text_input("url",
                            placeholder="https://www.youtube.com/watch?v=…  or direct .mp4 URL",
                            label_visibility="collapsed")
        analyze_url = st.button("▶  Scan URL", key="url_btn", disabled=not url)

        # ★ Clearly visible hint (uses .panel-hint class with proper colour)
        st.markdown("""
        <div class="panel-hint">
            ✅ &nbsp;YouTube · Instagram · X · Direct MP4 · Google Drive · Dropbox · Vimeo<br>
            <span style="font-size:0.65rem;opacity:0.75;">
                YouTube uses automatic Piped / Invidious proxy bypass
            </span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # RIGHT — Upload panel
    with right_col:
        st.markdown('<div class="neon-card neon-card-red">', unsafe_allow_html=True)
        st.markdown('<span class="badge badge-red">📁 Local File</span>', unsafe_allow_html=True)
        st.markdown('<div class="sec-header" style="font-size:0.85rem;color:var(--neon-red);">Upload &amp; Analyze</div>',
                    unsafe_allow_html=True)
        video_file   = st.file_uploader("video", type=["mp4","avi","mov","mkv"],
                                        label_visibility="collapsed")
        analyze_file = st.button("▶  Scan File", key="file_btn", disabled=(video_file is None))

        # ★ Clearly visible hint (uses .panel-hint.panel-hint-red class)
        st.markdown("""
        <div class="panel-hint panel-hint-red">
            ✅ &nbsp;Recommended — Works 100% of the time<br>
            MP4 · AVI · MOV · MKV — all formats accepted
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # URL analysis
    if analyze_url and url:
        with st.spinner("⟳ Connecting to proxy / downloading…"):
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    save_path   = os.path.join(tmpdir, "video.mp4")
                    actual_path = download_video(url, save_path)
                    with st.spinner("⟳ Running frame analysis…"):
                        label, conf, frame_probs = predict_video(
                            actual_path, model, threshold, frame_count)
                st.markdown("<br>", unsafe_allow_html=True)
                show_result(label, conf, frame_probs, threshold)
            except RuntimeError as e:
                st.error("Download failed")
                st.markdown(str(e))
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.info("💡 Download the video manually and use the **Upload File** panel instead.")

    # File analysis
    if analyze_file and video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="video-label">▶ VIDEO PREVIEW</div>', unsafe_allow_html=True)
        st.video(tmp_path)
        with st.spinner("⟳ Running frame analysis…"):
            label, conf, frame_probs = predict_video(tmp_path, model, threshold, frame_count)
        os.unlink(tmp_path)
        st.markdown("<br>", unsafe_allow_html=True)
        show_result(label, conf, frame_probs, threshold)

# ──────────────────────────────────────────
#  TAB 2: USER GUIDE
# ──────────────────────────────────────────
with tab_guide:
    gcol1, gcol2 = st.columns([1.1, 0.9], gap="large")

    with gcol1:
        st.markdown('<div class="sec-header">How to Use DeepShield</div>', unsafe_allow_html=True)
        for num, title, body in [
            ("01","Choose Input Method",
             "Paste a YouTube link or any direct .mp4 URL — the app auto-routes through a proxy. "
             "Or upload a local video file for 100% reliable analysis."),
            ("02","Configure Settings",
             "Expand 'Detection Settings' and tune Frame Count (5–30) and Threshold (0.3–0.7). "
             "Higher frame counts improve accuracy; lower thresholds flag more content as synthetic."),
            ("03","Submit Your Video",
             "Click Scan URL or Scan File. The engine downloads (if needed), extracts frames, "
             "and runs AI inference on each sampled frame."),
            ("04","Read the Verdict",
             "AUTHENTIC (green) = genuine footage with no manipulation detected. "
             "DEEPFAKE (red) = synthetic content detected with a confidence percentage."),
            ("05","Interpret the Graphs",
             "Gauge = overall confidence. Line chart = per-frame fake probability over time. "
             "Bar chart = per-frame risk. Histogram = score distribution across all frames."),
            ("06","Take Action",
             "If flagged as fake, cross-verify with the original source, report to the platform, "
             "and file a cyber complaint — see the Awareness & Help tab for helpline contacts."),
        ]:
            st.markdown(f"""
            <div class="guide-step">
                <div class="step-num">{num}</div>
                <div class="step-body"><h4>{title}</h4><p>{body}</p></div>
            </div>""", unsafe_allow_html=True)

    with gcol2:
        st.markdown('<div class="sec-header" style="color:var(--neon-red);">Quick Reference</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="neon-card neon-card-green">', unsafe_allow_html=True)
        st.markdown('<span class="badge badge-green">Model Specs</span>', unsafe_allow_html=True)
        st.markdown("""
        <table style="width:100%;font-family:'Share Tech Mono',monospace;font-size:0.72rem;border-collapse:collapse;">
            <tr><td style="padding:0.35rem 0;color:#2a4a3a;">ARCHITECTURE</td><td style="text-align:right;color:#4ade80;">MobileNetV2</td></tr>
            <tr style="border-top:1px solid rgba(57,255,20,0.1)"><td style="padding:0.35rem 0;color:#2a4a3a;">INPUT SIZE</td><td style="text-align:right;color:#4ade80;">128 × 128 px</td></tr>
            <tr style="border-top:1px solid rgba(57,255,20,0.1)"><td style="padding:0.35rem 0;color:#2a4a3a;">FRAME ACC.</td><td style="text-align:right;color:#4ade80;">88.8 %</td></tr>
            <tr style="border-top:1px solid rgba(57,255,20,0.1)"><td style="padding:0.35rem 0;color:#2a4a3a;">VIDEO ACC.</td><td style="text-align:right;color:#4ade80;">90.9 %</td></tr>
            <tr style="border-top:1px solid rgba(57,255,20,0.1)"><td style="padding:0.35rem 0;color:#2a4a3a;">ROC AUC</td><td style="text-align:right;color:#4ade80;">0.922</td></tr>
            <tr style="border-top:1px solid rgba(57,255,20,0.1)"><td style="padding:0.35rem 0;color:#2a4a3a;">DATASET</td><td style="text-align:right;color:#4ade80;">DFD</td></tr>
        </table>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="neon-card neon-card-red" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.markdown('<span class="badge badge-red">Tips for Best Results</span>', unsafe_allow_html=True)
        for icon, tip in [
            ("✅","Upload the file directly — most reliable method"),
            ("🎯","Use 720p+ videos for higher accuracy"),
            ("⏱","Minimum 5 seconds of footage recommended"),
            ("💡","Well-lit, front-facing videos score best"),
            ("🔄","Try threshold 0.45 for borderline cases"),
            ("📊","High score volatility often indicates editing"),
        ]:
            st.markdown(f"""<div style="display:flex;gap:0.7rem;align-items:center;margin-bottom:0.55rem;
                        font-family:'Rajdhani',sans-serif;font-size:0.88rem;color:#bb7777;">
                <span>{icon}</span><span>{tip}</span></div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────
#  TAB 3: AWARENESS & HELP
# ──────────────────────────────────────────
with tab_awareness:
    acol1, acol2 = st.columns([1,1], gap="large")

    with acol1:
        st.markdown('<div class="sec-header" style="color:var(--neon-red);">⚠ The Deepfake Threat</div>',
                    unsafe_allow_html=True)
        for title, body in [
            ("What Is a Deepfake?",
             "AI-generated synthetic media where a person's face or voice is replaced using GANs "
             "and autoencoders — nearly indistinguishable from authentic footage without forensics."),
            ("Why Is It Dangerous?",
             "Deepfakes enable misinformation campaigns, political manipulation, financial fraud, "
             "and non-consensual intimate imagery, spreading faster than the truth can catch up."),
            ("Scale of the Problem",
             "Over 500,000 deepfake videos circulated in 2023, doubling every six months. "
             "Detection tools, legislation, and media literacy are the three pillars of defence."),
        ]:
            st.markdown(f"""<div class="aware-card"><h4>{title}</h4><p>{body}</p></div>""",
                        unsafe_allow_html=True)

    with acol2:
        st.markdown('<div class="sec-header" style="color:var(--neon-amber);">🛡 Protect Yourself</div>',
                    unsafe_allow_html=True)
        for title, body in [
            ("Verify Before You Share",
             "Reverse-search the video, check metadata, and cross-reference trusted news sources "
             "before sharing anything extraordinary or emotionally charged."),
            ("Look for Visual Artefacts",
             "Unnatural blinking, mismatched lighting, blurry hairlines, and irregular skin "
             "texture are common tells. Slow to 0.25× speed to reveal frame glitches."),
            ("Use Multiple Detection Tools",
             "DeepShield, Sensity AI, and Microsoft Video Authenticator all provide forensic "
             "analysis. Use several detectors for high-stakes verification."),
        ]:
            st.markdown(f"""
            <div class="aware-card" style="background:rgba(255,180,0,0.03);border-color:rgba(255,180,0,0.14);">
                <h4 style="color:var(--neon-amber);">{title}</h4>
                <p style="color:#b89a60;">{body}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    pcol1, pcol2 = st.columns(2, gap="large")

    with pcol1:
        st.markdown('<div class="sec-header" style="color:var(--neon-cyan);">📋 Complaint Filing Process</div>',
                    unsafe_allow_html=True)
        for num, title, body in [
            ("01","Document Evidence",
             "Screenshot/record the content. Note the URL, date, time, and platform. "
             "Do NOT delete anything — evidence preservation is critical."),
            ("02","Report to Platform",
             "Use the platform's Report feature. Select 'Fake or deceptive content' / "
             "'Deepfake' / 'Synthetic media'."),
            ("03","File Online Cyber Complaint",
             "Visit cybercrime.gov.in → Report Cyber Crime. "
             "Attach screenshots, URLs, and all collected evidence."),
            ("04","Visit Nearest Cyber Police Station",
             "File an FIR with printed evidence and your online complaint registration number."),
            ("05","Follow Up",
             "Note your FIR/complaint number and follow up within 7–10 days. "
             "Escalate to your state cyber crime cell if no action is taken."),
        ]:
            st.markdown(f"""
            <div class="process-step">
                <div class="process-circle">{num}</div>
                <div class="process-content"><h4>{title}</h4><p>{body}</p></div>
            </div>""", unsafe_allow_html=True)

    with pcol2:
        st.markdown('<div class="sec-header" style="color:var(--neon-amber);">📞 Cyber Helplines</div>',
                    unsafe_allow_html=True)
        for icon, title, subtitle, desc, link in [
            ("🚨","National Cyber Crime Helpline",
             "Helpline: <strong style='color:#ffb400;font-size:1rem;'>1930</strong>",
             "Available 24×7 for reporting deepfakes, fraud, and online harassment.",
             "https://cybercrime.gov.in"),
            ("🌐","National Cyber Crime Reporting Portal",
             "cybercrime.gov.in",
             "File online complaints for social media crimes and financial cyber offences.",
             "https://cybercrime.gov.in"),
            ("👮","I4C — Indian Cyber Crime Coordination Centre",
             "Ministry of Home Affairs",
             "Nodal authority coordinating cyber crime investigation across India.",
             "https://i4c.mha.gov.in"),
            ("⚖️","Odisha Police Cyber Crime Cell",
             "Odisha Cyber Crime Unit",
             "State-level cyber crime reporting and investigation unit for Odisha.",
             "https://odishapolice.gov.in"),
            ("📧","CERT-In — Cyber Emergency Response",
             "incidents@cert-in.org.in | 1800-11-4949",
             "India's national agency for responding to cyber security incidents.",
             "https://www.cert-in.org.in"),
            ("⚠️","National Commission for Women",
             "Helpline: <strong style='color:#ffb400;'>7827170170</strong>",
             "For deepfake / non-consensual intimate imagery targeting women.",
             "http://ncw.nic.in"),
        ]:
            st.markdown(f"""
            <div class="helpline-card">
                <div class="helpline-icon">{icon}</div>
                <div class="helpline-info">
                    <h4>{title}</h4>
                    <p>{subtitle}</p>
                    <p style="font-family:'Rajdhani',sans-serif;font-size:0.82rem;
                              color:#907050;margin-top:0.25rem;">{desc}</p>
                    <a href="{link}" target="_blank">🔗 {link.replace('https://','').replace('http://','')}</a>
                </div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════
st.markdown("""
<div class="footer-bar">
    <span>⬡ DeepShield · Neural Forensics Platform · v2.0</span>
    <span>Frame Acc: 88.8% · Video Acc: 90.9% · AUC: 0.922</span>
    <span>DFD Dataset · Frame-level Analysis</span>
</div>""", unsafe_allow_html=True)