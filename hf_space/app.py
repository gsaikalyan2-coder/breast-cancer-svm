# hf_space/app.py — Standalone Streamlit for Hugging Face Spaces
# Loads pkl directly — no FastAPI needed

import streamlit as st
import joblib
import numpy as np
from PIL import Image
import time

st.set_page_config(
    page_title="Breast Cancer SVM Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("breast_cancer_svm.pkl")

model = load_model()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
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
p, span, div { color: #E8F4F2 !important; font-family: 'DM Sans', sans-serif !important; }
h1,h2,h3 { color: #E8F4F2 !important; font-family: 'DM Sans', sans-serif !important; font-weight: 700 !important; }
.stButton > button {
    background: #0D1520 !important; color: #7DA8B8 !important;
    border: 1.5px solid #234060 !important; border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
    font-size: 15px !important; padding: 14px 24px !important;
    transition: all 0.2s !important; width: 100% !important;
}
.stButton > button:hover {
    border-color: #35D4C0 !important; color: #E8F4F2 !important;
    transform: translateY(-2px) !important;
}
.stSlider > div > div > div { background: #234060 !important; }
.stSlider > div > div > div > div { background: #35D4C0 !important; }
[data-testid="stSlider"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important; color: #7DA8B8 !important;
    white-space: nowrap !important; overflow: hidden !important;
    text-overflow: ellipsis !important;
}
[data-testid="stExpander"] {
    background: #0D1520 !important; border: 1px solid #1E3045 !important;
    border-radius: 12px !important; margin-bottom: 12px !important;
}
[data-testid="stExpander"] summary {
    color: #35D4C0 !important; font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important; font-weight: 600 !important;
    letter-spacing: 0.07em !important; text-transform: uppercase !important;
    padding: 14px 16px !important;
}
[data-testid="stFileUploader"] {
    background: #0D1520 !important; border: 2px dashed #234060 !important;
    border-radius: 14px !important; padding: 20px !important;
}
hr { border-color: #1E3045 !important; margin: 20px 0 !important; }
.stSpinner > div { border-top-color: #35D4C0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Feature helpers ───────────────────────────────────────────────────────────
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
    b_mean = arr[:,:,2].mean()
    r_mean = arr[:,:,0].mean()
    if saturation > 110 and brightness > 180:
        return False
    if b_mean > r_mean * 1.6 and b_mean > 160:
        return False
    return True

# ── Result card ───────────────────────────────────────────────────────────────
def result_card(label, confidence, pm, pb):
    is_mal  = (label == "malignant")
    color   = "#D85A30" if is_mal else "#35D4C0"
    bg      = "linear-gradient(135deg,#2D0A0A,rgba(45,10,10,0.7))" if is_mal else "linear-gradient(135deg,#042E2A,rgba(4,46,42,0.7))"
    border  = "#D85A30" if is_mal else "#0E9E8C"
    icon    = "&#9888;" if is_mal else "&#10003;"
    label_u = label.upper()
    nuc_bg  = "rgba(216,90,48,.15)" if is_mal else "rgba(13,158,140,.15)"
    meaning = (
        "The tumor <strong style='color:#E8F4F2'>is cancerous</strong>. "
        "Malignant cells grow uncontrollably, invade nearby tissue, and can spread to other organs. "
        "<strong style='color:#D85A30'>Immediate medical consultation required.</strong>"
    ) if is_mal else (
        "The tumor <strong style='color:#E8F4F2'>is not cancerous</strong>. "
        "Benign cells are abnormal but confined — they do not invade surrounding tissue or spread. "
        "<strong style='color:#35D4C0'>Regular monitoring is still advised.</strong>"
    )
    html  = "<div style='background:" + bg + ";border:1.5px solid " + border + ";border-radius:16px;padding:26px;margin-top:16px'>"
    html += "<div style='display:flex;align-items:center;gap:14px;margin-bottom:18px'>"
    html += "<div style='width:54px;height:54px;border-radius:12px;flex-shrink:0;background:" + nuc_bg + ";border:1.5px solid " + border + ";display:flex;align-items:center;justify-content:center;font-size:26px'>" + icon + "</div>"
    html += "<div><div style='font-size:28px;font-weight:700;color:" + color + ";font-family:IBM Plex Mono,monospace;letter-spacing:.04em'>" + label_u + "</div>"
    html += "<div style='font-size:13px;color:#7DA8B8;font-family:IBM Plex Mono,monospace;margin-top:2px'>Confidence: <strong style='color:#E8F4F2;font-size:15px'>" + str(round(confidence,1)) + "%</strong></div></div></div>"
    html += "<div style='background:rgba(0,0,0,0.3);border-radius:10px;padding:14px 16px;margin-bottom:18px;border-left:3px solid " + border + "'>"
    html += "<div style='font-size:9px;font-family:IBM Plex Mono,monospace;color:#3E6070;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px'>What this means</div>"
    html += "<div style='font-size:13px;color:#B8D4D0;line-height:1.6;font-family:DM Sans,sans-serif'>" + meaning + "</div></div>"
    html += "<div style='margin-bottom:10px'><div style='display:flex;justify-content:space-between;font-size:11px;font-family:IBM Plex Mono,monospace;color:#7DA8B8;margin-bottom:5px'><span>Malignant probability</span><span style='color:#D85A30;font-weight:600'>" + str(round(pm,1)) + "%</span></div>"
    html += "<div style='background:#1E3045;border-radius:4px;height:8px;overflow:hidden'><div style='background:linear-gradient(90deg,#8B2500,#D85A30);width:" + str(int(pm)) + "%;height:100%;border-radius:4px'></div></div></div>"
    html += "<div style='margin-bottom:18px'><div style='display:flex;justify-content:space-between;font-size:11px;font-family:IBM Plex Mono,monospace;color:#7DA8B8;margin-bottom:5px'><span>Benign probability</span><span style='color:#35D4C0;font-weight:600'>" + str(round(pb,1)) + "%</span></div>"
    html += "<div style='background:#1E3045;border-radius:4px;height:8px;overflow:hidden'><div style='background:linear-gradient(90deg,#0A5C52,#35D4C0);width:" + str(int(pb)) + "%;height:100%;border-radius:4px'></div></div></div>"
    html += "<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:16px'>"
    for dl, dv, dc in [("Prediction",label_u,color),("Confidence",str(round(confidence,1))+"%","#35D4C0"),("Model","SVM · RBF","#7DA8B8")]:
        html += "<div style='background:#111C2B;border:1px solid #1E3045;border-radius:8px;padding:10px 12px'><div style='font-size:9px;color:#3E6070;font-family:IBM Plex Mono,monospace;text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px'>" + dl + "</div><div style='font-size:14px;font-family:IBM Plex Mono,monospace;font-weight:600;color:" + dc + "'>" + dv + "</div></div>"
    html += "</div><div style='font-size:11px;color:#3E6070;font-family:IBM Plex Mono,monospace;border-top:1px solid #1E3045;padding-top:12px;line-height:1.5'>&#9888; Educational purposes only — not a clinical diagnostic tool.</div></div>"
    return html

# ── Scan animation ────────────────────────────────────────────────────────────
SCAN_STEPS = [
    ("Initializing diagnostic sequence...",  8,  1.2),
    ("Mapping cellular boundaries...",       22,  1.0),
    ("Scanning tissue matrix...",            40,  1.4),
    ("Extracting morphological vectors...",  58,  1.2),
    ("Normalizing feature space...",         74,  0.9),
    ("Running SVM classifier...",            90,  0.7),
    ("Diagnosis complete.",                 100,  0.3),
]
STEP_ICONS = ["🔬","🧬","📡","⬡","📊","🤖","✅"]

def scan_frame(step_idx, bar, text):
    steps_html = ""
    for i,(t,_,_) in enumerate(SCAN_STEPS):
        if i < step_idx:
            dc,bc,marker = "#35D4C0","background:#35D4C0","✓"
        elif i == step_idx:
            dc,bc,marker = "#0E9E8C","background:#35D4C0;box-shadow:0 0 8px #35D4C0","◎"
        else:
            dc,bc,marker = "#3E6070","background:#1E3045","○"
        steps_html += "<div style='display:flex;align-items:center;gap:8px;font-size:11px;font-family:IBM Plex Mono,monospace;color:"+dc+";padding:3px 0'><div style='width:6px;height:6px;border-radius:50%;flex-shrink:0;"+bc+"'></div>"+marker+" "+STEP_ICONS[i]+" "+t+"</div>"
    seg_html = "<div style='display:flex;gap:3px;margin-bottom:16px;height:6px'>"
    for seg in range(1,8):
        if seg*14 <= bar:
            ss = "flex:1;border-radius:3px;height:6px;background:linear-gradient(90deg,#0A5C52,#35D4C0);box-shadow:0 0 6px rgba(53,212,192,0.6)"
        else:
            ss = "flex:1;border-radius:3px;height:6px;background:#1E3045"
        seg_html += "<div style='"+ss+"'></div>"
    seg_html += "</div>"
    spinner = "<span style='display:inline-block;font-size:14px;margin-right:6px'>◌</span>" if step_idx < len(SCAN_STEPS)-1 else ""
    hc = "#35D4C0" if step_idx==len(SCAN_STEPS)-1 else "#0E9E8C"
    html  = "<div style='background:#0D1520;border:1.5px solid #0E9E8C;border-radius:14px;padding:22px;position:relative;overflow:hidden'>"
    html += "<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:14px'>"
    html += "<div style='font-size:13px;color:#35D4C0;font-family:IBM Plex Mono,monospace;display:flex;align-items:center;gap:6px;font-weight:500'>"+spinner+"<strong>"+STEP_ICONS[min(step_idx,6)]+" "+text+"</strong></div>"
    html += "<div style='font-family:IBM Plex Mono,monospace;font-size:11px;color:"+hc+";border:1px solid "+hc+";border-radius:6px;padding:3px 10px'>"+str(bar)+"%</div></div>"
    html += seg_html
    html += "<div style='background:#080D12;border-radius:8px;padding:12px 14px;border:1px solid #1E3045'>"+steps_html+"</div></div>"
    return html

# ══════════════════════════════════════════════════════════════════════════════
# PAGE
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='background:#0D1520;border:1px solid #1E3045;border-radius:16px;"
    "padding:22px 28px;margin-bottom:24px;display:flex;align-items:center;gap:16px'>"
    "<div style='width:50px;height:50px;flex-shrink:0;background:linear-gradient(135deg,#0A5C52,#0E9E8C);"
    "border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:24px;"
    "box-shadow:0 0 20px rgba(14,158,140,0.5)'>&#128300;</div>"
    "<div style='flex:1'>"
    "<div style='font-family:IBM Plex Mono,monospace;font-size:10px;color:#35D4C0;"
    "letter-spacing:.12em;text-transform:uppercase;margin-bottom:4px'>SVM · Wisconsin Dataset · v2.1</div>"
    "<div style='font-size:22px;font-weight:700;color:#E8F4F2'>Breast Cancer Diagnostic Interface</div>"
    "<div style='font-size:12px;color:#7DA8B8;margin-top:2px'>Select a mode below to begin analysis</div></div>"
    "<div style='font-family:IBM Plex Mono,monospace;font-size:10px;background:#042E2A;color:#35D4C0;"
    "border:1px solid #0E9E8C;border-radius:20px;padding:6px 14px'>&#9679; MODEL LOADED</div></div>",
    unsafe_allow_html=True
)

# ── Mode buttons ──────────────────────────────────────────────────────────────
st.markdown(
    "<div style='font-family:IBM Plex Mono,monospace;font-size:10px;color:#3E6070;"
    "text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px'>Select analysis mode</div>",
    unsafe_allow_html=True
)

mc1, mc2 = st.columns(2)
with mc1:
    btn_params = st.button("🔬  Verify Parameters\nManually adjust 30 clinical measurements", key="bp")
with mc2:
    btn_scan   = st.button("🩻  Quick Scan\nUpload a tumor image for instant analysis",      key="bs")

if btn_params: st.session_state["mode"] = "params"
if btn_scan:   st.session_state["mode"] = "scan"
mode = st.session_state.get("mode", None)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODE A — VERIFY PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
if mode == "params":
    ex = st.session_state.get("example_vals", None)
    def dv(i,fb): return float(ex[i]) if ex else fb

    with st.expander("📊  GROUP 1 — MEAN VALUES  [features 1–10]", expanded=True):
        r1 = st.columns(5)
        mean_radius     = r1[0].slider("mean radius",   6.98,  28.11, dv(0,14.13),  key="a0")
        mean_texture    = r1[1].slider("mean texture",  9.71,  39.28, dv(1,19.29),  key="a1")
        mean_perimeter  = r1[2].slider("mean perim",   43.79, 188.50, dv(2,91.97),  key="a2")
        mean_area       = r1[3].slider("mean area",   143.50,2501.00, dv(3,654.89), key="a3")
        mean_smooth     = r1[4].slider("mean smooth",  0.053,  0.163, dv(4,0.096),  key="a4")
        r2 = st.columns(5)
        mean_compact    = r2[0].slider("mean compact", 0.019,  0.345, dv(5,0.104),  key="a5")
        mean_concav     = r2[1].slider("mean concav",  0.000,  0.427, dv(6,0.089),  key="a6")
        mean_conc_pts   = r2[2].slider("mean c pts",   0.000,  0.201, dv(7,0.049),  key="a7")
        mean_symm       = r2[3].slider("mean symm",    0.106,  0.304, dv(8,0.181),  key="a8")
        mean_fractal    = r2[4].slider("mean fractal", 0.050,  0.097, dv(9,0.063),  key="a9")

    with st.expander("📉  GROUP 2 — STANDARD ERROR  [features 11–20]", expanded=False):
        r3 = st.columns(5)
        radius_se    = r3[0].slider("radius SE",   0.112, 2.873, dv(10,0.405), key="a10")
        texture_se   = r3[1].slider("texture SE",  0.360, 4.885, dv(11,1.217), key="a11")
        perim_se     = r3[2].slider("perim SE",    0.757,21.980, dv(12,2.866), key="a12")
        area_se      = r3[3].slider("area SE",     6.802,542.20, dv(13,40.34), key="a13")
        smooth_se    = r3[4].slider("smooth SE",   0.002, 0.031, dv(14,0.007), key="a14")
        r4 = st.columns(5)
        compact_se   = r4[0].slider("compact SE",  0.002, 0.135, dv(15,0.025), key="a15")
        concav_se    = r4[1].slider("concav SE",   0.000, 0.396, dv(16,0.032), key="a16")
        conc_pts_se  = r4[2].slider("c pts SE",    0.000, 0.053, dv(17,0.012), key="a17")
        symm_se      = r4[3].slider("symm SE",     0.008, 0.079, dv(18,0.021), key="a18")
        fractal_se   = r4[4].slider("fractal SE",  0.001, 0.030, dv(19,0.004), key="a19")

    with st.expander("📈  GROUP 3 — WORST VALUES  [features 21–30]", expanded=False):
        r5 = st.columns(5)
        w_radius  = r5[0].slider("worst radius",  7.930, 36.04, dv(20,16.27),  key="a20")
        w_texture = r5[1].slider("worst texture",12.020, 49.54, dv(21,25.68),  key="a21")
        w_perim   = r5[2].slider("worst perim",  50.410,251.20, dv(22,107.26), key="a22")
        w_area    = r5[3].slider("worst area",  185.200,4254.0, dv(23,880.58), key="a23")
        w_smooth  = r5[4].slider("worst smooth",  0.071, 0.223, dv(24,0.132),  key="a24")
        r6 = st.columns(5)
        w_compact = r6[0].slider("worst compact", 0.027, 1.058, dv(25,0.254),  key="a25")
        w_concav  = r6[1].slider("worst concav",  0.000, 1.252, dv(26,0.272),  key="a26")
        w_c_pts   = r6[2].slider("worst c pts",   0.000, 0.291, dv(27,0.115),  key="a27")
        w_symm    = r6[3].slider("worst symm",    0.156, 0.664, dv(28,0.290),  key="a28")
        w_fractal = r6[4].slider("worst fractal", 0.055, 0.208, dv(29,0.084),  key="a29")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    _, bc, _ = st.columns([1,2,1])
    with bc:
        do_analyze = st.button("🔬  Analyze Tumor", key="analyze", use_container_width=True, type="primary")

    if do_analyze:
        values = [mean_radius,mean_texture,mean_perimeter,mean_area,mean_smooth,
                  mean_compact,mean_concav,mean_conc_pts,mean_symm,mean_fractal,
                  radius_se,texture_se,perim_se,area_se,smooth_se,
                  compact_se,concav_se,conc_pts_se,symm_se,fractal_se,
                  w_radius,w_texture,w_perim,w_area,w_smooth,
                  w_compact,w_concav,w_c_pts,w_symm,w_fractal]
        arr   = np.array(values).reshape(1,-1)
        pred  = model.predict(arr)[0]
        proba = model.predict_proba(arr)[0]
        label = "benign" if pred == 1 else "malignant"
        st.markdown(result_card(label, proba.max()*100,
                    proba[0]*100, proba[1]*100), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODE B — QUICK SCAN
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "scan":
    uploaded = st.file_uploader("Drop tumor image — JPG or PNG", type=["jpg","jpeg","png"])

    if uploaded:
        img_col, ctrl_col = st.columns([1.2,1])
        with img_col:
            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img, use_container_width=True, caption="Uploaded image")
        with ctrl_col:
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            run_scan = st.button("🔬  Run Diagnostic Scan", use_container_width=True, type="primary", key="rs")

        if run_scan:
            if not is_likely_medical(pil_img):
                st.markdown(
                    "<div style='background:#1A0A0A;border:1.5px solid #D85A30;"
                    "border-radius:12px;padding:28px;margin-top:16px;text-align:center'>"
                    "<div style='font-size:36px;margin-bottom:12px'>&#9888;</div>"
                    "<div style='font-size:17px;font-weight:700;color:#D85A30;"
                    "font-family:IBM Plex Mono,monospace;margin-bottom:10px'>INVALID IMAGE</div>"
                    "<div style='font-size:13px;color:#B87060;font-family:DM Sans,sans-serif;line-height:1.7'>"
                    "Please upload a <strong style='color:#E8F4F2'>histopathology slide</strong> "
                    "or <strong style='color:#E8F4F2'>ultrasound image</strong> for analysis.</div>"
                    "<div style='margin-top:14px;font-size:10px;font-family:IBM Plex Mono,monospace;color:#3E6070'>"
                    "Valid: tissue biopsies · ultrasound · mammography · microscopy slides</div></div>",
                    unsafe_allow_html=True
                )
            else:
                scan_ph   = st.empty()
                result_ph = st.empty()
                for i,(text,bar,wait) in enumerate(SCAN_STEPS):
                    scan_ph.markdown(scan_frame(i,bar,text), unsafe_allow_html=True)
                    time.sleep(wait)
                feats = image_to_30_features(pil_img)
                pred  = model.predict(feats)[0]
                proba = model.predict_proba(feats)[0]
                label = "benign" if pred==1 else "malignant"
                scan_ph.markdown(scan_frame(len(SCAN_STEPS)-1,100,"Diagnosis complete."), unsafe_allow_html=True)
                result_ph.markdown(result_card(label, proba.max()*100,
                                   proba[0]*100, proba[1]*100), unsafe_allow_html=True)

else:
    st.markdown(
        "<div style='background:#0D1520;border:1px dashed #234060;border-radius:14px;"
        "padding:40px;text-align:center;margin-top:8px'>"
        "<div style='font-size:32px;margin-bottom:12px'>🔬</div>"
        "<div style='font-size:15px;font-weight:600;color:#7DA8B8;margin-bottom:6px'>"
        "Select a mode above to begin</div>"
        "<div style='font-size:11px;color:#3E6070;font-family:IBM Plex Mono,monospace'>"
        "Verify Parameters &#8594; manual sliders &nbsp;·&nbsp; Quick Scan &#8594; image upload</div></div>",
        unsafe_allow_html=True
    )

st.markdown(
    "<div style='font-family:IBM Plex Mono,monospace;font-size:10px;color:#3E6070;"
    "text-align:center;margin-top:40px;padding-top:16px;border-top:1px solid #1E3045'>"
    "Wisconsin Diagnostic Breast Cancer · sklearn SVC · RBF kernel · "
    "<span style='color:#35D4C0'>569 samples</span> · "
    "<span style='color:#35D4C0'>30 features</span> · AUC ~0.99 · &#9888; Educational only</div>",
    unsafe_allow_html=True
)