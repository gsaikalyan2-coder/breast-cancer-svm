# frontend/streamlit_app.py — v3.1 Clean CSS fix

import streamlit as st
import requests
import numpy as np
from PIL import Image
import time

st.set_page_config(
    page_title="Breast Cancer SVM Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

API_URL = "http://localhost:8000/predict"

# ── Inject CSS — no comments inside style block ──────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=DM+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main {
    background-color: #080D12 !important;
    color: #E8F4F2 !important;
    font-family: 'DM Sans', sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stAppViewContainer"] > .main > div {
    padding-top: 24px !important;
    padding-left: 40px !important;
    padding-right: 40px !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
}
p, span, div {
    color: #E8F4F2 !important;
    font-family: 'DM Sans', sans-serif !important;
}
h1, h2, h3 {
    color: #E8F4F2 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
}
.stButton > button {
    background: #0D1520 !important;
    color: #7DA8B8 !important;
    border: 1.5px solid #234060 !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 14px 24px !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    border-color: #35D4C0 !important;
    color: #E8F4F2 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(14,158,140,0.25) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#0A5C52,#0E9E8C) !important;
    color: #E8F4F2 !important;
    border: none !important;
    box-shadow: 0 4px 20px rgba(14,158,140,0.4) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(14,158,140,0.55) !important;
}
.stSlider > div > div > div { background: #234060 !important; }
.stSlider > div > div > div > div { background: #35D4C0 !important; }
[data-testid="stSlider"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
    color: #7DA8B8 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
[data-testid="stExpander"] {
    background: #0D1520 !important;
    border: 1px solid #1E3045 !important;
    border-radius: 12px !important;
    margin-bottom: 12px !important;
}
[data-testid="stExpander"] summary {
    color: #35D4C0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    padding: 14px 16px !important;
}
[data-testid="stExpander"] > div > div { padding: 16px !important; }
[data-testid="stFileUploader"] {
    background: #0D1520 !important;
    border: 2px dashed #234060 !important;
    border-radius: 14px !important;
    padding: 20px !important;
}
[data-testid="stFileUploader"]:hover { border-color: #35D4C0 !important; }
[data-testid="stFileUploader"] label {
    color: #7DA8B8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stMetric"] {
    background: #0D1520 !important;
    border: 1px solid #1E3045 !important;
    border-radius: 10px !important;
    padding: 14px !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
    color: #3E6070 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #35D4C0 !important;
}
hr { border-color: #1E3045 !important; margin: 20px 0 !important; }
[data-testid="stHorizontalBlock"] { gap: 14px !important; }
.stSpinner > div { border-top-color: #35D4C0 !important; }
@keyframes p { 0%,100%{opacity:1} 50%{opacity:.3} }
@keyframes fadeIn { from{opacity:0;transform:scale(.97)} to{opacity:1;transform:scale(1)} }
@keyframes sweep { 0%{top:0;opacity:1} 85%{top:100%;opacity:.8} 100%{top:100%;opacity:0} }
@keyframes spin { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
</style>
""", unsafe_allow_html=True)


# ── Feature ranges ────────────────────────────────────────────────────────────
FEAT_MINS = [6.98,9.71,43.79,143.5,0.053,0.019,0.0,0.0,0.106,0.05,
             0.112,0.36,0.757,6.802,0.002,0.002,0.0,0.0,0.008,0.001,
             7.93,12.02,50.41,185.2,0.071,0.027,0.0,0.0,0.156,0.055]
FEAT_MAXS = [28.11,39.28,188.5,2501,0.163,0.345,0.427,0.201,0.304,0.097,
             2.873,4.885,21.98,542.2,0.031,0.135,0.396,0.053,0.079,0.030,
             36.04,49.54,251.2,4254,0.223,1.058,1.252,0.291,0.664,0.208]

def image_to_30_features(pil_img):
    img  = pil_img.convert("RGB").resize((64, 64))
    arr  = np.array(img, dtype=np.float32)
    gray = arr.mean(axis=2)
    ch_stds = arr.std(axis=(0, 1))
    ch_maxs = arr.max(axis=(0, 1))
    gm, gs  = gray.mean(), gray.std()
    def scale(raw, rmin, rmax, idx):
        n = np.clip((raw - rmin) / (rmax - rmin + 1e-6), 0, 1)
        return FEAT_MINS[idx] + n * (FEAT_MAXS[idx] - FEAT_MINS[idx])
    feats = np.zeros(30)
    for i in range(10):
        feats[i]    = scale(gm + i * gs * 0.1, 0, 255, i)
    for i in range(10):
        feats[10+i] = scale(ch_stds[i % 3], 0, 80, 10+i)
    for i in range(10):
        feats[20+i] = scale(ch_maxs[i % 3], 0, 255, 20+i)
    return feats.reshape(1, -1)

def is_likely_medical(pil_img):
    img = pil_img.convert("RGB").resize((32, 32))
    arr = np.array(img, dtype=np.float32)
    saturation = (arr.max(axis=2) - arr.min(axis=2)).mean()
    brightness = arr.mean()
    b_mean = float(arr[:,:,2].mean())
    r_mean = float(arr[:,:,0].mean())
    if saturation > 110 and brightness > 180:
        return False
    if b_mean > r_mean * 1.6 and b_mean > 160:
        return False
    return True


# ── Result card ───────────────────────────────────────────────────────────────
def result_card(label, confidence, pm, pb):
    is_mal  = label == "malignant"
    color   = "#D85A30" if is_mal else "#35D4C0"
    bg      = "linear-gradient(135deg,#2D0A0A,rgba(45,10,10,0.7))" if is_mal else "linear-gradient(135deg,#042E2A,rgba(4,46,42,0.7))"
    border  = "#D85A30" if is_mal else "#0E9E8C"
    icon    = "&#9888;" if is_mal else "&#10003;"
    lu      = label.upper()
    meaning = (
        "The tumor <strong style='color:#E8F4F2'>is cancerous</strong>. "
        "Malignant cells grow uncontrollably, invade nearby tissue, and can spread. "
        "<strong style='color:#D85A30'>Immediate medical consultation required.</strong>"
    ) if is_mal else (
        "The tumor <strong style='color:#E8F4F2'>is not cancerous</strong>. "
        "Benign cells are abnormal but confined and do not spread. "
        "<strong style='color:#35D4C0'>Regular monitoring is still advised.</strong>"
    )
    cells = "".join([
        f'<div style="background:#111C2B;border:1px solid #1E3045;border-radius:8px;padding:10px 12px">'
        f'<div style="font-size:9px;color:#3E6070;font-family:IBM Plex Mono,monospace;text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px">{dl}</div>'
        f'<div style="font-size:14px;font-family:IBM Plex Mono,monospace;font-weight:600;color:{dc}">{dv}</div></div>'
        for dl,dv,dc in [("Prediction",lu,color),("Confidence",f"{confidence:.1f}%","#35D4C0"),("Model","SVM · RBF","#7DA8B8")]
    ])
    return f"""
<div style="background:{bg};border:1.5px solid {border};border-radius:16px;padding:26px;margin-top:16px;animation:fadeIn .5s ease">
  <div style="display:flex;align-items:center;gap:14px;margin-bottom:18px">
    <div style="width:54px;height:54px;border-radius:12px;flex-shrink:0;background:{'rgba(216,90,48,.15)' if is_mal else 'rgba(13,158,140,.15)'};border:1.5px solid {border};display:flex;align-items:center;justify-content:center;font-size:26px">{icon}</div>
    <div>
      <div style="font-size:28px;font-weight:700;color:{color};font-family:'IBM Plex Mono',monospace;letter-spacing:.04em">{lu}</div>
      <div style="font-size:13px;color:#7DA8B8;font-family:'IBM Plex Mono',monospace;margin-top:2px">Confidence: <strong style="color:#E8F4F2;font-size:15px">{confidence:.1f}%</strong></div>
    </div>
  </div>
  <div style="background:rgba(0,0,0,0.3);border-radius:10px;padding:14px 16px;margin-bottom:18px;border-left:3px solid {border}">
    <div style="font-size:9px;font-family:'IBM Plex Mono',monospace;color:#3E6070;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px">What this means</div>
    <div style="font-size:13px;color:#B8D4D0;line-height:1.6;font-family:'DM Sans',sans-serif">{meaning}</div>
  </div>
  <div style="margin-bottom:10px">
    <div style="display:flex;justify-content:space-between;font-size:11px;font-family:'IBM Plex Mono',monospace;color:#7DA8B8;margin-bottom:5px">
      <span>Malignant probability</span><span style="color:#D85A30;font-weight:600">{pm:.1f}%</span>
    </div>
    <div style="background:#1E3045;border-radius:4px;height:8px;overflow:hidden">
      <div style="background:linear-gradient(90deg,#8B2500,#D85A30);width:{int(pm)}%;height:100%;border-radius:4px"></div>
    </div>
  </div>
  <div style="margin-bottom:18px">
    <div style="display:flex;justify-content:space-between;font-size:11px;font-family:'IBM Plex Mono',monospace;color:#7DA8B8;margin-bottom:5px">
      <span>Benign probability</span><span style="color:#35D4C0;font-weight:600">{pb:.1f}%</span>
    </div>
    <div style="background:#1E3045;border-radius:4px;height:8px;overflow:hidden">
      <div style="background:linear-gradient(90deg,#0A5C52,#35D4C0);width:{int(pb)}%;height:100%;border-radius:4px"></div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:16px">{cells}</div>
  <div style="font-size:11px;color:#3E6070;font-family:'IBM Plex Mono',monospace;border-top:1px solid #1E3045;padding-top:12px;display:flex;align-items:flex-start;gap:6px;line-height:1.5">
    <span>&#9888;</span><span>For educational purposes only. Always consult a qualified medical professional.</span>
  </div>
</div>"""


# ── Scan animation frame ──────────────────────────────────────────────────────
SCAN_STEPS = [
    ("Initializing diagnostic sequence",  8,  1.2),
    ("Mapping cellular boundaries",       22,  1.0),
    ("Scanning tissue matrix",            40,  1.4),
    ("Extracting morphological vectors",  58,  1.2),
    ("Normalizing feature space",         74,  0.9),
    ("Running SVM classifier",            90,  0.7),
    ("Diagnosis complete",               100,  0.3),
]
STEP_ICONS = ["🔬","🧬","📡","⬡","📊","🤖","✅"]

def scan_frame(step_idx, bar, show_sweep=False):
    sweep_css = (
        "position:absolute;left:0;right:0;height:2px;"
        "background:linear-gradient(90deg,transparent,#35D4C0,transparent);"
        "box-shadow:0 0 12px #35D4C0;animation:sweep 1.6s ease-in-out;top:0;"
    ) if show_sweep else "display:none"

    steps_html = ""
    for i,(t,_,_) in enumerate(SCAN_STEPS):
        if i < step_idx:
            dc,bc,mk = "#35D4C0","background:#35D4C0","&#10003;"
        elif i == step_idx:
            dc,bc,mk = "#35D4C0","background:#35D4C0;box-shadow:0 0 8px #35D4C0","&#9676;"
        else:
            dc,bc,mk = "#3E6070","background:#1E3045","&#9675;"
        steps_html += (
            f'<div style="display:flex;align-items:center;gap:8px;font-size:11px;'
            f'font-family:IBM Plex Mono,monospace;color:{dc};padding:3px 0">'
            f'<div style="width:6px;height:6px;border-radius:50%;flex-shrink:0;{bc}"></div>'
            f'{mk} {STEP_ICONS[i]} {t}</div>'
        )

    spinner_html = (
        '<span style="display:inline-block;animation:spin .6s linear infinite;font-size:14px;margin-right:6px">&#9675;</span>'
        if step_idx < len(SCAN_STEPS)-1 else "&#10003; "
    )

    txt  = SCAN_STEPS[step_idx][0]
    segs = "".join([
        f'<div style="flex:1;border-radius:3px;height:6px;'
        f'background:{"linear-gradient(90deg,#0A5C52,#35D4C0)" if (s+1)*14<=bar else "#1E3045"};'
        f'box-shadow:{"0 0 6px rgba(53,212,192,0.6)" if (s+1)*14<=bar else "none"};'
        f'transition:background .3s"></div>'
        for s in range(7)
    ])

    return f"""
<div style="background:#0D1520;border:1.5px solid #0E9E8C;border-radius:14px;padding:22px;position:relative;overflow:hidden;box-shadow:0 0 32px rgba(14,158,140,0.15)">
  <div style="{sweep_css}"></div>
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px">
    <div style="font-size:13px;color:#35D4C0;font-family:'IBM Plex Mono',monospace;display:flex;align-items:center;gap:6px;font-weight:500">
      {spinner_html}<strong>{txt}</strong>
    </div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#35D4C0;border:1px solid #0E9E8C;border-radius:6px;padding:3px 10px;background:rgba(14,158,140,0.08)">{bar}%</div>
  </div>
  <div style="display:flex;gap:3px;margin-bottom:16px;height:6px">{segs}</div>
  <div style="background:#080D12;border-radius:8px;padding:12px 14px;border:1px solid #1E3045">{steps_html}</div>
  <div style="position:absolute;bottom:10px;right:14px;font-family:'IBM Plex Mono',monospace;font-size:9px;color:#1E3045;letter-spacing:.1em">SVM v2.1</div>
</div>"""


# ══════════════════════════════════════════════════════════════════════════════
# RENDER
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#0D1520;border:1px solid #1E3045;border-radius:16px;padding:22px 28px;margin-bottom:24px;display:flex;align-items:center;gap:16px;box-shadow:0 4px 24px rgba(0,0,0,0.4)">
  <div style="width:50px;height:50px;flex-shrink:0;background:linear-gradient(135deg,#0A5C52,#0E9E8C);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:24px;box-shadow:0 0 20px rgba(14,158,140,0.5)">&#128300;</div>
  <div style="flex:1">
    <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#35D4C0;letter-spacing:.12em;text-transform:uppercase;margin-bottom:4px">SVM · Wisconsin Dataset · v2.1</div>
    <div style="font-size:22px;font-weight:700;color:#E8F4F2;font-family:'DM Sans',sans-serif">Breast Cancer Diagnostic Interface</div>
    <div style="font-size:12px;color:#7DA8B8;margin-top:2px;font-family:'DM Sans',sans-serif">Select a mode below to begin analysis</div>
  </div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;background:#042E2A;color:#35D4C0;border:1px solid #0E9E8C;border-radius:20px;padding:6px 14px;white-space:nowrap;display:flex;align-items:center;gap:7px">
    <span style="width:7px;height:7px;border-radius:50%;background:#35D4C0;display:inline-block;animation:p 2s infinite"></span> MODEL LOADED
  </div>
</div>
""", unsafe_allow_html=True)


# ── Malignant vs Benign explanation ───────────────────────────────────────────
mal_features = ["Irregular, jagged cell borders","Large and varied nucleus size","High concavity and compactness","Spreads beyond tissue walls","Worst radius typically above 20mm"]
ben_features = ["Smooth, rounded cell borders","Uniform and smaller nucleus size","Low concavity and compactness","Remains within tissue walls","Worst radius typically below 16mm"]

mal_feat_html = "".join([
    f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;border-bottom:1px solid rgba(216,90,48,0.15);color:#B87060;font-size:10px;font-family:IBM Plex Mono,monospace">'
    f'<span style="color:#D85A30;font-size:12px">&#8594;</span>{f}</div>'
    for f in mal_features
])
ben_feat_html = "".join([
    f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;border-bottom:1px solid rgba(13,158,140,0.15);color:#5BA89A;font-size:10px;font-family:IBM Plex Mono,monospace">'
    f'<span style="color:#35D4C0;font-size:12px">&#8594;</span>{f}</div>'
    for f in ben_features
])

st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:28px">

  <div style="background:linear-gradient(135deg,#1A0505,#2D0A0A);border:1.5px solid #D85A30;border-radius:14px;padding:20px">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">
      <span style="font-size:20px">&#9888;</span>
      <span style="font-size:17px;font-weight:700;color:#D85A30;font-family:'IBM Plex Mono',monospace;letter-spacing:.04em">MALIGNANT</span>
      <span style="font-size:10px;font-family:'IBM Plex Mono',monospace;background:rgba(216,90,48,.15);color:#D85A30;border:1px solid #D85A30;border-radius:4px;padding:2px 7px;margin-left:auto">LABEL 0 · HIGH RISK</span>
    </div>
    <svg width="100%" height="90" viewBox="0 0 280 90" xmlns="http://www.w3.org/2000/svg" style="margin-bottom:12px;border-radius:8px;background:#140303;display:block">
      <path d="M40,28 L58,14 L76,22 L84,40 L72,56 L50,60 L33,50 Z" fill="rgba(216,90,48,0.2)" stroke="#D85A30" stroke-width="1.5"/>
      <path d="M58,14 L70,5 L84,12 L88,26 L76,22 Z" fill="rgba(216,90,48,0.1)" stroke="#D85A30" stroke-width="1" stroke-dasharray="2 1"/>
      <circle cx="58" cy="37" r="6" fill="#D85A30" opacity="0.8"/>
      <circle cx="62" cy="41" r="3.5" fill="#8B2500"/>
      <path d="M112,18 L132,12 L150,26 L154,48 L138,60 L114,55 L100,42 L106,26 Z" fill="rgba(216,90,48,0.2)" stroke="#D85A30" stroke-width="1.5"/>
      <path d="M150,26 L168,20 L172,36 L160,44 L154,48 Z" fill="rgba(216,90,48,0.1)" stroke="#D85A30" stroke-width="1" stroke-dasharray="2 1"/>
      <circle cx="130" cy="37" r="8" fill="#D85A30" opacity="0.7"/>
      <circle cx="134" cy="41" r="4" fill="#8B2500"/>
      <path d="M188,22 L208,16 L224,28 L228,50 L212,62 L190,58 L178,44 Z" fill="rgba(216,90,48,0.2)" stroke="#D85A30" stroke-width="1.5"/>
      <circle cx="205" cy="40" r="7" fill="#D85A30" opacity="0.75"/>
      <circle cx="208" cy="44" r="3.5" fill="#8B2500"/>
      <path d="M84,40 L100,40" stroke="#D85A30" stroke-width="1.2" stroke-dasharray="3 2" opacity="0.7"/>
      <path d="M154,40 L178,40" stroke="#D85A30" stroke-width="1.2" stroke-dasharray="3 2" opacity="0.7"/>
      <polygon points="100,37 106,40 100,43" fill="#D85A30" opacity="0.7"/>
      <polygon points="178,37 184,40 178,43" fill="#D85A30" opacity="0.7"/>
      <text x="8" y="84" font-family="IBM Plex Mono" font-size="8" fill="#5A2010">irregular · varied size · invades tissue</text>
    </svg>
    <div style="font-size:12px;color:#E8C4B8;line-height:1.65;font-family:'DM Sans',sans-serif;margin-bottom:12px">
      Cancerous tumor. Cells are structurally <strong style="color:#E8F4F2">abnormal and irregular</strong>, divide uncontrollably, and break through tissue boundaries to <strong style="color:#D85A30">invade neighboring organs</strong>.
    </div>
    {mal_feat_html}
  </div>

  <div style="background:linear-gradient(135deg,#020E0C,#042E2A);border:1.5px solid #0E9E8C;border-radius:14px;padding:20px">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">
      <span style="font-size:20px">&#10003;</span>
      <span style="font-size:17px;font-weight:700;color:#35D4C0;font-family:'IBM Plex Mono',monospace;letter-spacing:.04em">BENIGN</span>
      <span style="font-size:10px;font-family:'IBM Plex Mono',monospace;background:rgba(13,158,140,.15);color:#35D4C0;border:1px solid #0E9E8C;border-radius:4px;padding:2px 7px;margin-left:auto">LABEL 1 · LOWER RISK</span>
    </div>
    <svg width="100%" height="90" viewBox="0 0 280 90" xmlns="http://www.w3.org/2000/svg" style="margin-bottom:12px;border-radius:8px;background:#020C09;display:block">
      <circle cx="46"  cy="44" r="22" fill="rgba(13,158,140,0.12)" stroke="#0E9E8C" stroke-width="1.5"/>
      <circle cx="46"  cy="44" r="8"  fill="#0E9E8C" opacity="0.65"/>
      <circle cx="49"  cy="41" r="3"  fill="#35D4C0" opacity="0.9"/>
      <circle cx="106" cy="44" r="20" fill="rgba(13,158,140,0.12)" stroke="#0E9E8C" stroke-width="1.5"/>
      <circle cx="106" cy="44" r="7"  fill="#0E9E8C" opacity="0.65"/>
      <circle cx="109" cy="41" r="2.5" fill="#35D4C0" opacity="0.9"/>
      <circle cx="166" cy="44" r="21" fill="rgba(13,158,140,0.12)" stroke="#0E9E8C" stroke-width="1.5"/>
      <circle cx="166" cy="44" r="8"  fill="#0E9E8C" opacity="0.65"/>
      <circle cx="169" cy="41" r="3"  fill="#35D4C0" opacity="0.9"/>
      <circle cx="226" cy="44" r="19" fill="rgba(13,158,140,0.12)" stroke="#0E9E8C" stroke-width="1.5"/>
      <circle cx="226" cy="44" r="7"  fill="#0E9E8C" opacity="0.65"/>
      <circle cx="229" cy="41" r="2.5" fill="#35D4C0" opacity="0.9"/>
      <rect x="14" y="12" width="252" height="64" rx="6" fill="none" stroke="#35D4C0" stroke-width="0.8" stroke-dasharray="4 3" opacity="0.25"/>
      <text x="8" y="84" font-family="IBM Plex Mono" font-size="8" fill="#1A6B5E">smooth border · uniform size · stays contained</text>
    </svg>
    <div style="font-size:12px;color:#B8D4D0;line-height:1.65;font-family:'DM Sans',sans-serif;margin-bottom:12px">
      Non-cancerous tumor. Cells are <strong style="color:#E8F4F2">round and uniform</strong>, grow slowly, and remain contained within their original tissue boundary. <strong style="color:#35D4C0">Does not spread to other organs.</strong>
    </div>
    {ben_feat_html}
  </div>

</div>
""", unsafe_allow_html=True)


# ── Mode selector ─────────────────────────────────────────────────────────────
st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:10px;color:#3E6070;text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px">Select analysis mode</div>', unsafe_allow_html=True)

mc1, mc2 = st.columns(2)
with mc1:
    btn_params = st.button("🔬  Verify Parameters\nManually adjust 30 clinical measurements", key="btn_params")
with mc2:
    btn_scan_mode = st.button("🩻  Quick Scan\nUpload a tumor image for instant analysis", key="btn_scan")

if btn_params:
    st.session_state["mode"] = "params"
if btn_scan_mode:
    st.session_state["mode"] = "scan"

mode = st.session_state.get("mode", None)
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE A — VERIFY PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
if mode == "params":
    st.markdown("""
<div style="background:#0D1520;border:1px solid #234060;border-radius:12px;padding:14px 20px;margin-bottom:20px;display:flex;align-items:center;gap:10px">
  <span style="font-size:16px">&#128300;</span>
  <div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#35D4C0;font-weight:600">VERIFY PARAMETERS MODE</div>
    <div style="font-size:12px;color:#7DA8B8;margin-top:2px">Adjust all 30 cell nucleus measurements using the sliders below</div>
  </div>
</div>""", unsafe_allow_html=True)

    ex = st.session_state.get("example_vals", None)
    def dv(i, fb): return float(ex[i]) if ex else fb

    with st.expander("GROUP 1  —  MEAN VALUES  [ features 1 to 10 ]", expanded=True):
        r1 = st.columns(5)
        mean_radius  = r1[0].slider("mean radius",  6.98,  28.11, dv(0, 14.13), key="p0")
        mean_texture = r1[1].slider("mean texture", 9.71,  39.28, dv(1, 19.29), key="p1")
        mean_perim   = r1[2].slider("mean perim",  43.79, 188.50, dv(2, 91.97), key="p2")
        mean_area    = r1[3].slider("mean area",  143.50,2501.00, dv(3,654.89), key="p3")
        mean_smooth  = r1[4].slider("mean smooth",  0.053,  0.163, dv(4,  0.096),key="p4")
        r2 = st.columns(5)
        mean_compact = r2[0].slider("mean compact", 0.019, 0.345, dv(5,  0.104),key="p5")
        mean_concav  = r2[1].slider("mean concav",  0.000, 0.427, dv(6,  0.089),key="p6")
        mean_cpts    = r2[2].slider("mean c-pts",   0.000, 0.201, dv(7,  0.049),key="p7")
        mean_symm    = r2[3].slider("mean symm",    0.106, 0.304, dv(8,  0.181),key="p8")
        mean_fract   = r2[4].slider("mean fract",   0.050, 0.097, dv(9,  0.063),key="p9")

    with st.expander("GROUP 2  —  STANDARD ERROR  [ features 11 to 20 ]", expanded=False):
        r3 = st.columns(5)
        radius_se    = r3[0].slider("radius SE",    0.112, 2.873, dv(10, 0.405),key="p10")
        texture_se   = r3[1].slider("texture SE",   0.360, 4.885, dv(11, 1.217),key="p11")
        perim_se     = r3[2].slider("perim SE",     0.757,21.980, dv(12, 2.866),key="p12")
        area_se      = r3[3].slider("area SE",      6.802,542.20, dv(13,40.340),key="p13")
        smooth_se    = r3[4].slider("smooth SE",    0.002, 0.031, dv(14, 0.007),key="p14")
        r4 = st.columns(5)
        compact_se   = r4[0].slider("compact SE",   0.002, 0.135, dv(15, 0.025),key="p15")
        concav_se    = r4[1].slider("concav SE",    0.000, 0.396, dv(16, 0.032),key="p16")
        cpts_se      = r4[2].slider("c-pts SE",     0.000, 0.053, dv(17, 0.012),key="p17")
        symm_se      = r4[3].slider("symm SE",      0.008, 0.079, dv(18, 0.021),key="p18")
        fract_se     = r4[4].slider("fract SE",     0.001, 0.030, dv(19, 0.004),key="p19")

    with st.expander("GROUP 3  —  WORST VALUES  [ features 21 to 30 ]", expanded=False):
        r5 = st.columns(5)
        worst_rad    = r5[0].slider("worst rad",    7.930, 36.04, dv(20,16.270),key="p20")
        worst_tex    = r5[1].slider("worst tex",   12.020, 49.54, dv(21,25.680),key="p21")
        worst_per    = r5[2].slider("worst per",   50.410,251.20, dv(22,107.26),key="p22")
        worst_area   = r5[3].slider("worst area", 185.200,4254.0, dv(23,880.58),key="p23")
        worst_smo    = r5[4].slider("worst smo",    0.071, 0.223, dv(24,  0.132),key="p24")
        r6 = st.columns(5)
        worst_com    = r6[0].slider("worst com",    0.027, 1.058, dv(25,  0.254),key="p25")
        worst_con    = r6[1].slider("worst con",    0.000, 1.252, dv(26,  0.272),key="p26")
        worst_cpt    = r6[2].slider("worst c-pt",   0.000, 0.291, dv(27,  0.115),key="p27")
        worst_sym    = r6[3].slider("worst sym",    0.156, 0.664, dv(28,  0.290),key="p28")
        worst_fra    = r6[4].slider("worst fra",    0.055, 0.208, dv(29,  0.084),key="p29")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    _, bc, _ = st.columns([1,2,1])
    with bc:
        do_analyze = st.button("Analyze Tumor", key="analyze_btn", use_container_width=True, type="primary")

    if do_analyze:
        payload = {
            "mean_radius":mean_radius,"mean_texture":mean_texture,
            "mean_perimeter":mean_perim,"mean_area":mean_area,
            "mean_smoothness":mean_smooth,"mean_compactness":mean_compact,
            "mean_concavity":mean_concav,"mean_concave_points":mean_cpts,
            "mean_symmetry":mean_symm,"mean_fractal_dimension":mean_fract,
            "radius_error":radius_se,"texture_error":texture_se,
            "perimeter_error":perim_se,"area_error":area_se,
            "smoothness_error":smooth_se,"compactness_error":compact_se,
            "concavity_error":concav_se,"concave_points_error":cpts_se,
            "symmetry_error":symm_se,"fractal_dimension_error":fract_se,
            "worst_radius":worst_rad,"worst_texture":worst_tex,
            "worst_perimeter":worst_per,"worst_area":worst_area,
            "worst_smoothness":worst_smo,"worst_compactness":worst_com,
            "worst_concavity":worst_con,"worst_concave_points":worst_cpt,
            "worst_symmetry":worst_sym,"worst_fractal_dimension":worst_fra,
        }
        with st.spinner("Running SVM classifier..."):
            try:
                r = requests.post(API_URL, json=payload, timeout=10)
                res = r.json()
                st.markdown(result_card(res["label"],res["confidence"],
                    res["probabilities"]["malignant"],res["probabilities"]["benign"]),
                    unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Cannot reach FastAPI: {e}")
                st.code("cd C:\\Users\\saika\\breast-cancer-svm\\backend\npython -m uvicorn app:app --reload --port 8000",language="powershell")


# ══════════════════════════════════════════════════════════════════════════════
# MODE B — QUICK SCAN
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "scan":
    st.markdown("""
<div style="background:#0D1520;border:1px solid #234060;border-radius:12px;padding:14px 20px;margin-bottom:20px;display:flex;align-items:center;gap:10px">
  <span style="font-size:16px">&#129467;</span>
  <div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#35D4C0;font-weight:600">QUICK SCAN MODE</div>
    <div style="font-size:12px;color:#7DA8B8;margin-top:2px">Upload a histopathology or ultrasound image for SVM analysis</div>
  </div>
</div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop tumor image here — JPG or PNG", type=["jpg","jpeg","png"])

    if uploaded:
        img_col, ctrl_col = st.columns([1.2, 1])
        with img_col:
            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img, use_container_width=True, caption="Uploaded image")
        with ctrl_col:
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            run_scan = st.button("Run Diagnostic Scan", use_container_width=True, type="primary")

        if run_scan:
            if not is_likely_medical(pil_img):
                st.markdown("""
<div style="background:#1A0A0A;border:1.5px solid #D85A30;border-radius:12px;padding:26px;margin-top:16px;text-align:center">
  <div style="font-size:36px;margin-bottom:12px">&#9888;</div>
  <div style="font-size:17px;font-weight:700;color:#D85A30;font-family:'IBM Plex Mono',monospace;margin-bottom:10px">INVALID IMAGE</div>
  <div style="font-size:13px;color:#B87060;font-family:'DM Sans',sans-serif;line-height:1.6">
    This image does not appear to be a medical scan.<br>
    Please upload a <strong style="color:#E8F4F2">histopathology slide</strong> or <strong style="color:#E8F4F2">ultrasound image</strong>.
  </div>
  <div style="margin-top:14px;font-size:10px;font-family:'IBM Plex Mono',monospace;color:#3E6070">
    Valid inputs: tissue biopsies · ultrasound scans · mammography · microscopy slides
  </div>
</div>""", unsafe_allow_html=True)
            else:
                sp  = st.empty()
                rp  = st.empty()
                for i,(text,bar,wait) in enumerate(SCAN_STEPS):
                    sp.markdown(scan_frame(i, bar, i==2), unsafe_allow_html=True)
                    time.sleep(wait)

                feats = image_to_30_features(pil_img)
                payload = {k:float(feats[0,i]) for i,k in enumerate([
                    "mean_radius","mean_texture","mean_perimeter","mean_area",
                    "mean_smoothness","mean_compactness","mean_concavity",
                    "mean_concave_points","mean_symmetry","mean_fractal_dimension",
                    "radius_error","texture_error","perimeter_error","area_error",
                    "smoothness_error","compactness_error","concavity_error",
                    "concave_points_error","symmetry_error","fractal_dimension_error",
                    "worst_radius","worst_texture","worst_perimeter","worst_area",
                    "worst_smoothness","worst_compactness","worst_concavity",
                    "worst_concave_points","worst_symmetry","worst_fractal_dimension",
                ])}
                try:
                    r = requests.post(API_URL, json=payload, timeout=10)
                    res = r.json()
                    sp.markdown(scan_frame(len(SCAN_STEPS)-1, 100, False), unsafe_allow_html=True)
                    rp.markdown(result_card(res["label"],res["confidence"],
                        res["probabilities"]["malignant"],res["probabilities"]["benign"]),
                        unsafe_allow_html=True)
                except Exception as e:
                    sp.empty()
                    st.error(f"Cannot reach FastAPI: {e}")
                    st.code("cd C:\\Users\\saika\\breast-cancer-svm\\backend\npython -m uvicorn app:app --reload --port 8000",language="powershell")

else:
    st.markdown("""
<div style="background:#0D1520;border:1px dashed #234060;border-radius:14px;padding:48px;text-align:center;margin-top:8px">
  <div style="font-size:36px;margin-bottom:14px">&#128300;</div>
  <div style="font-size:16px;font-weight:600;color:#7DA8B8;font-family:'DM Sans',sans-serif;margin-bottom:6px">Select a mode above to begin</div>
  <div style="font-size:12px;color:#3E6070;font-family:'IBM Plex Mono',monospace">
    Verify Parameters — manual slider control &nbsp;·&nbsp; Quick Scan — image upload
  </div>
</div>""", unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#3E6070;text-align:center;margin-top:40px;padding-top:16px;border-top:1px solid #1E3045">
  Wisconsin Diagnostic Breast Cancer · sklearn SVC · RBF kernel ·
  <span style="color:#35D4C0">569 samples</span> ·
  <span style="color:#35D4C0">30 features</span> ·
  AUC ~0.99 &nbsp;·&nbsp; Educational only
</div>
""", unsafe_allow_html=True)