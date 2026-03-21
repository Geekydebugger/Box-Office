# ============================================================
#   CINEPREDICT — app.py  (v6 — Cinematic Redesign)
#   Run with:  streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, sqlite3, datetime
from difflib import get_close_matches

DB_PATH    = "models/box_office.db"
GENRE_LIST = ["Action","Adventure","Biography","Comedy","Crime","Drama",
              "Family","Fantasy","Historical","Horror","Musical","Mystery",
              "Romance","Sci-Fi","Sports","Supernatural","Thriller"]

def verdict_from_profit(p):
    if p < 0:     return "FLOP"
    elif p < 50:  return "AVERAGE"
    elif p < 100: return "HIT"
    elif p < 200: return "SUPER HIT"
    else:          return "BLOCKBUSTER"

def get_season(month):
    if month in [1,10,11,12]: return "Holiday"
    elif month in [3,4,5]:    return "Summer"
    elif month in [6,7]:      return "Monsoon"
    else:                      return "Normal"

VERDICT_STYLES = {
    "BLOCKBUSTER": {"color":"#FFD700","glow":"rgba(255,215,0,0.5)","bg":"rgba(255,215,0,0.07)","emoji":"🏆","desc":"200%+ profit"},
    "SUPER HIT":   {"color":"#00E676","glow":"rgba(0,230,118,0.5)","bg":"rgba(0,230,118,0.07)","emoji":"⭐","desc":"100–200% profit"},
    "HIT":         {"color":"#40C4FF","glow":"rgba(64,196,255,0.5)","bg":"rgba(64,196,255,0.07)","emoji":"✅","desc":"50–100% profit"},
    "AVERAGE":     {"color":"#B0BEC5","glow":"rgba(176,190,197,0.3)","bg":"rgba(176,190,197,0.05)","emoji":"➡️","desc":"Break even"},
    "FLOP":        {"color":"#FF5252","glow":"rgba(255,82,82,0.5)","bg":"rgba(255,82,82,0.07)","emoji":"📉","desc":"Loss making"},
}

st.set_page_config(page_title="CinePredict", page_icon="🎬", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&family=Space+Mono:wght@400;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* ── App background ── */
.stApp {
    background: #07070f !important;
    font-family: 'DM Sans', sans-serif !important;
}
.main .block-container {
    padding: 0 1.5rem 4rem !important;
    max-width: 860px !important;
}
#MainMenu, footer, header, .stDeployButton { visibility: hidden !important; display: none !important; }

/* ── Animated background orbs ── */
.orb-container {
    position: fixed; inset: 0; pointer-events: none; z-index: 0; overflow: hidden;
}
.orb {
    position: absolute; border-radius: 50%; filter: blur(80px);
    animation: float 8s ease-in-out infinite;
}
.orb-1 {
    width: 500px; height: 500px; top: -150px; left: -100px;
    background: radial-gradient(circle, rgba(127,119,221,0.15), transparent 70%);
    animation-delay: 0s;
}
.orb-2 {
    width: 400px; height: 400px; bottom: -100px; right: -50px;
    background: radial-gradient(circle, rgba(216,90,48,0.1), transparent 70%);
    animation-delay: -3s;
}
.orb-3 {
    width: 300px; height: 300px; top: 40%; left: 50%;
    background: radial-gradient(circle, rgba(0,200,150,0.05), transparent 70%);
    animation-delay: -5s;
}
@keyframes float {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33%       { transform: translate(20px, -30px) scale(1.05); }
    66%       { transform: translate(-15px, 20px) scale(0.95); }
}

/* ── Film strip decoration ── */
.filmstrip {
    position: fixed; top: 0; left: 0; width: 18px; height: 100vh;
    background: rgba(255,255,255,0.02);
    border-right: 1px solid rgba(255,255,255,0.03);
    z-index: 0;
}
.filmstrip-hole {
    width: 8px; height: 12px;
    background: rgba(255,255,255,0.04);
    border-radius: 2px;
    margin: 6px auto;
}

/* ── Main content z-index ── */
.main, .stApp > div { position: relative; z-index: 1; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3.5rem 0 2rem;
    animation: fadeDown 0.8s cubic-bezier(0.16,1,0.3,1) forwards;
}
@keyframes fadeDown {
    from { opacity: 0; transform: translateY(-24px); }
    to   { opacity: 1; transform: translateY(0); }
}
.hero-tag {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem; letter-spacing: 0.35em;
    color: #7F77DD; text-transform: uppercase;
    background: rgba(127,119,221,0.08);
    border: 1px solid rgba(127,119,221,0.2);
    border-radius: 30px; padding: 0.3rem 1rem;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(4rem, 12vw, 7.5rem);
    line-height: 0.88;
    background: linear-gradient(160deg, #ffffff 0%, #c8c4ff 45%, #D85A30 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: 0.03em;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem; font-weight: 300;
    color: rgba(255,255,255,0.28);
    margin-top: 1rem; letter-spacing: 0.03em;
    line-height: 1.6;
}
.hero-divider {
    display: flex; align-items: center; gap: 1rem;
    margin: 2rem auto; max-width: 300px;
}
.hero-divider-line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(127,119,221,0.3), transparent);
}
.hero-divider-dot {
    width: 4px; height: 4px; border-radius: 50%;
    background: #7F77DD;
    box-shadow: 0 0 8px rgba(127,119,221,0.8);
}

/* ── Form section ── */
.form-wrap {
    animation: fadeUp 0.7s cubic-bezier(0.16,1,0.3,1) 0.15s both;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(28px); }
    to   { opacity: 1; transform: translateY(0); }
}
.form-section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem; letter-spacing: 0.3em;
    color: rgba(255,255,255,0.2); text-transform: uppercase;
    margin: 1.8rem 0 1rem;
    display: flex; align-items: center; gap: 0.8rem;
}
.form-section-title::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(255,255,255,0.06), transparent);
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    padding: 0.65rem 1rem !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: rgba(127,119,221,0.45) !important;
    box-shadow: 0 0 0 3px rgba(127,119,221,0.08), 0 0 20px rgba(127,119,221,0.05) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder { color: rgba(255,255,255,0.2) !important; }
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    color: #fff !important;
    transition: border-color 0.25s ease !important;
}
div[data-baseweb="select"] > div:focus-within {
    border-color: rgba(127,119,221,0.45) !important;
}
div[data-baseweb="select"] svg { fill: rgba(255,255,255,0.3) !important; }

/* Labels */
.stTextInput label, .stNumberInput label,
.stSelectbox label, .stCheckbox label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    color: rgba(255,255,255,0.35) !important;
    text-transform: uppercase !important;
    margin-bottom: 0.3rem !important;
}

/* Checkbox */
.stCheckbox > label {
    display: flex !important; align-items: center !important;
    gap: 0.5rem !important; cursor: pointer !important;
    padding: 0.7rem 1rem !important;
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
}
.stCheckbox > label:hover {
    background: rgba(127,119,221,0.06) !important;
    border-color: rgba(127,119,221,0.2) !important;
}

/* ── Predict button ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #7F77DD 0%, #6058C8 50%, #4A44A8 100%) !important;
    border: none !important;
    border-radius: 14px !important;
    color: #fff !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.25rem !important;
    letter-spacing: 0.18em !important;
    padding: 1rem 2rem !important;
    margin-top: 1.5rem !important;
    cursor: pointer !important;
    position: relative !important;
    overflow: hidden !important;
    transition: transform 0.25s cubic-bezier(0.34,1.56,0.64,1),
                box-shadow 0.25s ease !important;
}
.stButton > button::before {
    content: '' !important;
    position: absolute !important; inset: 0 !important;
    background: linear-gradient(135deg, rgba(255,255,255,0.1), transparent) !important;
    opacity: 0 !important; transition: opacity 0.25s ease !important;
}
.stButton > button:hover {
    transform: translateY(-3px) scale(1.01) !important;
    box-shadow: 0 12px 40px rgba(127,119,221,0.45),
                0 4px 15px rgba(127,119,221,0.3) !important;
}
.stButton > button:hover::before { opacity: 1 !important; }
.stButton > button:active {
    transform: translateY(-1px) scale(0.99) !important;
    transition: transform 0.1s ease !important;
}

/* ── Result area ── */
.result-wrap {
    animation: revealResult 0.6s cubic-bezier(0.16,1,0.3,1) forwards;
}
@keyframes revealResult {
    from { opacity: 0; transform: translateY(32px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

/* ── Result card ── */
.result-card {
    border-radius: 24px;
    padding: 3rem 2.5rem 2rem;
    text-align: center;
    position: relative; overflow: hidden;
    margin: 2rem 0 1rem;
}
.result-card-inner {
    position: relative; z-index: 2;
}
.result-card-bg {
    position: absolute; inset: 0; border-radius: 24px;
    transition: all 0.5s ease;
}
.result-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem; letter-spacing: 0.35em;
    text-transform: uppercase; opacity: 0.4;
    margin-bottom: 0.5rem;
}
.result-amount {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(4rem, 12vw, 7rem);
    line-height: 1; letter-spacing: 0.02em;
    color: #FFD700;
    filter: drop-shadow(0 0 30px rgba(255,215,0,0.4));
}
.result-range {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem; font-style: italic;
    opacity: 0.35; margin-top: 0.2rem;
}
.result-roi {
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem; margin-top: 0.6rem; opacity: 0.6;
}
.verdict-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    padding: 0.55rem 2rem;
    border-radius: 50px;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.35rem; letter-spacing: 0.12em;
    margin-top: 1.4rem;
    border: 1px solid;
    position: relative;
    animation: badgePop 0.5s cubic-bezier(0.34,1.56,0.64,1) 0.3s both;
}
@keyframes badgePop {
    from { opacity: 0; transform: scale(0.7); }
    to   { opacity: 1; transform: scale(1); }
}
.result-meta {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem; letter-spacing: 0.05em;
    opacity: 0.25; margin-top: 1.2rem;
}
.confidence-text {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem; margin-top: 0.5rem;
}

/* ── Range bar ── */
.range-section {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 14px; padding: 1.2rem 1.4rem;
    margin: 1rem 0;
}
.range-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.8rem;
}
.range-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem; letter-spacing: 0.2em;
    text-transform: uppercase; color: rgba(255,255,255,0.2);
}
.range-coverage {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem; color: rgba(255,255,255,0.2);
}
.range-track {
    height: 6px; background: rgba(255,255,255,0.05);
    border-radius: 3px; position: relative; overflow: visible;
}
.range-fill {
    position: absolute; top: 0; height: 100%;
    border-radius: 3px; opacity: 0.6;
    animation: expandRange 0.8s cubic-bezier(0.16,1,0.3,1) 0.4s both;
}
@keyframes expandRange {
    from { width: 0 !important; opacity: 0; }
}
.range-dot {
    position: absolute; top: 50%; transform: translate(-50%,-50%);
    width: 16px; height: 16px; border-radius: 50%;
    background: #FFD700; border: 2px solid #07070f;
    box-shadow: 0 0 14px rgba(255,215,0,0.7);
    animation: dotAppear 0.4s ease 0.9s both;
}
@keyframes dotAppear {
    from { opacity: 0; transform: translate(-50%,-50%) scale(0); }
    to   { opacity: 1; transform: translate(-50%,-50%) scale(1); }
}
.range-minmax {
    display: flex; justify-content: space-between;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem; color: rgba(255,255,255,0.2);
    margin-top: 0.5rem;
}

/* ── Metrics grid ── */
.metrics-grid {
    display: grid; grid-template-columns: repeat(4,1fr);
    gap: 8px; margin: 1rem 0;
}
.metric-box {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 0.9rem 0.7rem;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s;
    animation: fadeUp 0.5s cubic-bezier(0.16,1,0.3,1) both;
}
.metric-box:hover {
    border-color: rgba(127,119,221,0.2);
    transform: translateY(-2px);
}
.metric-box:nth-child(1) { animation-delay: 0.1s; }
.metric-box:nth-child(2) { animation-delay: 0.15s; }
.metric-box:nth-child(3) { animation-delay: 0.2s; }
.metric-box:nth-child(4) { animation-delay: 0.25s; }
.metric-lbl {
    font-family: 'Space Mono', monospace;
    font-size: 0.55rem; letter-spacing: 0.2em;
    text-transform: uppercase; color: rgba(255,255,255,0.25);
    margin-bottom: 0.35rem;
}
.metric-val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.35rem; color: #fff; line-height: 1;
}
.metric-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.68rem; margin-top: 0.15rem;
}

/* ── Stage boxes ── */
.stage-row {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 8px; margin: 1rem 0;
}
.stage-box {
    background: rgba(127,119,221,0.04);
    border: 1px solid rgba(127,119,221,0.1);
    border-left: 3px solid rgba(127,119,221,0.4);
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    animation: fadeUp 0.5s cubic-bezier(0.16,1,0.3,1) 0.35s both;
}
.stage-num {
    font-family: 'Space Mono', monospace;
    font-size: 0.55rem; letter-spacing: 0.2em;
    color: #7F77DD; text-transform: uppercase; margin-bottom: 0.3rem;
}
.stage-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem; font-weight: 500; color: rgba(255,255,255,0.6);
    margin-bottom: 0.2rem;
}
.stage-val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.4rem; color: #FFD700; line-height: 1;
}
.stage-desc {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.68rem; color: rgba(255,255,255,0.25);
    margin-top: 0.1rem;
}

/* ── Vote section ── */
.votes-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 1rem 0;
}
.votes-col-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem; letter-spacing: 0.2em;
    text-transform: uppercase; color: rgba(255,255,255,0.2);
    margin-bottom: 0.5rem;
}
.vote-item {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.45rem 0.8rem; border-radius: 8px;
    background: rgba(255,255,255,0.02);
    font-family: 'DM Sans', sans-serif; font-size: 0.78rem;
    color: rgba(255,255,255,0.45); margin: 3px 0;
    transition: background 0.15s;
}
.vote-item:hover { background: rgba(255,255,255,0.04); }
.vote-item.final {
    background: rgba(127,119,221,0.07);
    border: 1px solid rgba(127,119,221,0.18);
    color: #fff; font-weight: 500;
}
.vote-val { font-weight: 600; color: #FFD700; }
.vote-verdict { font-weight: 500; }

/* ── Verdict guide ── */
.vg-wrap {
    display: grid; grid-template-columns: repeat(5,1fr);
    gap: 6px; margin: 1.2rem 0 0;
}
.vg-item {
    text-align: center; padding: 0.7rem 0.3rem;
    border-radius: 10px; border: 1px solid;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    cursor: default;
}
.vg-item:hover {
    transform: translateY(-3px);
}
.vg-emoji { font-size: 1.1rem; }
.vg-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.78rem; letter-spacing: 0.04em;
    margin-top: 0.2rem;
}
.vg-pct {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.58rem; color: rgba(255,255,255,0.25);
    margin-top: 0.1rem;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
    margin-top: 0.8rem !important;
}
[data-testid="stExpander"] summary {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
    color: rgba(255,255,255,0.35) !important;
    padding: 0.8rem 1rem !important;
}

/* ── Footer ── */
.cp-footer {
    text-align: center; margin-top: 4rem; padding-top: 2rem;
    border-top: 1px solid rgba(255,255,255,0.04);
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem; letter-spacing: 0.25em;
    color: rgba(255,255,255,0.08); text-transform: uppercase;
    line-height: 2;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(127,119,221,0.3); border-radius: 2px; }
</style>

<!-- Animated background -->
<div class="orb-container">
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>
    <div class="orb orb-3"></div>
</div>
<div class="filmstrip">
    <div class="filmstrip-hole"></div><div class="filmstrip-hole"></div>
    <div class="filmstrip-hole"></div><div class="filmstrip-hole"></div>
    <div class="filmstrip-hole"></div><div class="filmstrip-hole"></div>
    <div class="filmstrip-hole"></div><div class="filmstrip-hole"></div>
    <div class="filmstrip-hole"></div><div class="filmstrip-hole"></div>
    <div class="filmstrip-hole"></div><div class="filmstrip-hole"></div>
    <div class="filmstrip-hole"></div><div class="filmstrip-hole"></div>
    <div class="filmstrip-hole"></div><div class="filmstrip-hole"></div>
    <div class="filmstrip-hole"></div><div class="filmstrip-hole"></div>
    <div class="filmstrip-hole"></div><div class="filmstrip-hole"></div>
</div>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    d = "models"
    required = ["regressor.pkl","classifier.pkl","stage1_model.pkl",
                "lower_bound.pkl","upper_bound.pkl",
                "label_encoder.pkl","label_language.pkl","label_season.pkl","label_genre.pkl",
                "star_power_map.pkl","director_power_map.pkl","meta.pkl","scaler_s2.pkl"]
    missing = [f for f in required if not os.path.exists(os.path.join(d,f))]
    if missing:
        return None, missing
    return {
        "regressor":          joblib.load(f"{d}/regressor.pkl"),
        "classifier":         joblib.load(f"{d}/classifier.pkl"),
        "stage1_model":       joblib.load(f"{d}/stage1_model.pkl"),
        "lower_bound":        joblib.load(f"{d}/lower_bound.pkl"),
        "upper_bound":        joblib.load(f"{d}/upper_bound.pkl"),
        "label_encoder":      joblib.load(f"{d}/label_encoder.pkl"),
        "label_language":     joblib.load(f"{d}/label_language.pkl"),
        "label_season":       joblib.load(f"{d}/label_season.pkl"),
        "label_genre":        joblib.load(f"{d}/label_genre.pkl"),
        "star_power_map":     joblib.load(f"{d}/star_power_map.pkl"),
        "director_power_map": joblib.load(f"{d}/director_power_map.pkl"),
        "meta":               joblib.load(f"{d}/meta.pkl"),
        "scaler_s2":          joblib.load(f"{d}/scaler_s2.pkl"),
    }, []

def save_prediction(p):
    if not os.path.exists(DB_PATH): return
    con = sqlite3.connect(DB_PATH)
    try:
        con.execute("""INSERT INTO predictions
            (predicted_at,movie_name,star,director,language,genre,budget,opening_day,
             screens,release_month,release_year,is_franchise,
             pred_worldwide,pred_lower,pred_upper,pred_opening_wk,
             pred_profit,pred_profit_pct,pred_verdict,clf_verdict,
             confidence,ensemble_agreement,star_power_used,dir_power_used)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            datetime.datetime.now().isoformat(),
            p["movie_name"],p["star"],p["director"],p["language"],p["genre"],
            p["budget"],p["opening_day"],p["screens"],6,2025,
            int(p["is_franchise"]),p["pred_worldwide"],p["pred_lower"],p["pred_upper"],
            p["pred_opening_wk"],p["pred_profit"],p["pred_profit_pct"],
            p["pred_verdict"],p["clf_verdict"],p["confidence"],
            int(p["ensemble_agreement"]),p["star_power_used"],p["dir_power_used"]
        ))
        con.commit()
    except Exception: pass
    con.close()

models, missing = load_models()
if models is None:
    st.error(f"⚠️ Models not found. Run `python main.py` first.\nMissing: {missing}")
    st.stop()

m        = models
meta     = m["meta"]
star_map = m["star_power_map"]
dir_map  = m["director_power_map"]
lang_enc = m["label_language"]
sea_enc  = m["label_season"]
gen_enc  = m["label_genre"]
top3_reg = meta.get("top3_reg", ["XGBoost","RandomForest","GradientBoosting"])
known_stars     = sorted(star_map.index.tolist())
known_directors = sorted(dir_map.index.tolist())

# ── Hero ──────────────────────────────────────────────────────
r2_pct  = int(meta.get('reg_r2',0)*100)
mae_val = int(meta.get('reg_mae_cr',0))
acc_pct = int(meta.get('clf_accuracy',0)*100)

st.markdown(f"""
<div class="hero">
    <div class="hero-tag">🎬 Indian Cinema Intelligence</div>
    <div class="hero-title">CinePredict</div>
    <div class="hero-sub">
        Predict lifetime box office collection for any Indian film<br>
        <span style="opacity:0.5">{r2_pct}% R² accuracy &nbsp;·&nbsp; ₹{mae_val}Cr avg error &nbsp;·&nbsp; {acc_pct}% verdict accuracy &nbsp;·&nbsp; 12-algo ensemble</span>
    </div>
    <div class="hero-divider">
        <div class="hero-divider-line"></div>
        <div class="hero-divider-dot"></div>
        <div class="hero-divider-line"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Form ──────────────────────────────────────────────────────
st.markdown('<div class="form-wrap">', unsafe_allow_html=True)

st.markdown('<div class="form-section-title">Film Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    movie_name    = st.text_input("Movie Title", placeholder="e.g. Tiger 4")
    star_name     = st.text_input("Lead Star", placeholder="e.g. Salman Khan",
                                  help=f"Known stars: {', '.join(known_stars[:5])}...")
    budget        = st.number_input("Budget (₹ Crore)", min_value=1.0, max_value=2000.0, value=100.0, step=5.0)

with col2:
    director_name = st.text_input("Director", placeholder="e.g. Rohit Shetty",
                                  help=f"Known directors: {', '.join(known_directors[:5])}...")
    language      = st.selectbox("Language", options=sorted(lang_enc.classes_.tolist()),
                                 index=list(lang_enc.classes_).index("Hindi")
                                 if "Hindi" in lang_enc.classes_ else 0)
    opening_day   = st.number_input("Opening Day Est. (₹ Crore)", min_value=0.1, max_value=500.0, value=20.0, step=1.0)

st.markdown('<div class="form-section-title">Release Details</div>', unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)
with col3:
    screens = st.number_input("Worldwide Screens", min_value=100, max_value=15000, value=4000, step=100)
with col4:
    genre   = st.selectbox("Genre", options=GENRE_LIST, index=GENRE_LIST.index("Action"))
with col5:
    is_franchise = st.checkbox("🔁 Sequel / Franchise film")

predict_btn = st.button("⚡  PREDICT BOX OFFICE", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── Prediction ────────────────────────────────────────────────
if predict_btn:
    if not movie_name.strip():
        st.warning("Please enter a movie title.")
        st.stop()

    star_match = get_close_matches(star_name.strip().title(), star_map.index.tolist(), n=1, cutoff=0.6)
    star_val   = float(star_map[star_match[0]]) if star_match else float(meta["global_star_mean"])
    star_note  = f"Matched **{star_match[0]}**" if star_match else "Using average star power"

    dir_match  = get_close_matches(director_name.strip().title(), dir_map.index.tolist(), n=1, cutoff=0.6)
    dir_val    = float(dir_map[dir_match[0]]) if dir_match else float(meta["global_director_mean"])
    dir_note   = f"Matched **{dir_match[0]}**" if dir_match else "Using average director power"

    lang_val   = int(lang_enc.transform([language])[0]) if language in lang_enc.classes_ else 0
    season_str = get_season(6)
    season_val = int(sea_enc.transform([season_str])[0]) if season_str in sea_enc.classes_ else 0
    genre_val  = int(gen_enc.transform([genre])[0]) if genre in gen_enc.classes_ else 0
    fest_val   = 0
    log_bud    = np.log1p(budget)
    bud_sq     = float(budget) ** 2
    rating     = 6.5  # default

    s1_row = pd.DataFrame([{
        "Budget":budget,"Screens":screens,"Language_Label":lang_val,
        "Season_Label":season_val,"Franchise":int(is_franchise),
        "Screens_to_Budget":screens/budget,"Release_Year":2025,
        "Genre_Label":genre_val,"Log_Budget":log_bud,"Budget_Squared":bud_sq,
        "Rating":rating,"Festival_Release":fest_val,
        "Star_Power":star_val,"Director_Power":dir_val,
    }])
    pred_opening_wk = round(float(np.expm1(m["stage1_model"].predict(s1_row)[0])), 1)

    s2_row = pd.DataFrame([{
        "Budget":float(budget),"Opening_Day":float(opening_day),"Screens":float(screens),
        "Language_Label":lang_val,"Season_Label":season_val,"Franchise":int(is_franchise),
        "Opening_to_Budget":float(opening_day)/float(budget),
        "Screens_to_Budget":float(screens)/float(budget),
        "Opening_per_Screen":float(opening_day)/float(screens) if screens>0 else 0,
        "Release_Year":2025,"Genre_Label":genre_val,
        "Rating":rating,"Log_Budget":log_bud,"Budget_Squared":bud_sq,
        "Overseas_Ratio":0.15,"Rating_x_Budget":rating*float(budget),
        "Festival_Release":fest_val,"Star_Power":star_val,"Director_Power":dir_val,
        "Pred_Opening_Week":pred_opening_wk,"Log_Opening_Week":np.log1p(pred_opening_wk),
    }])

    pred_log       = m["regressor"].predict(s2_row)[0]
    pred_worldwide = round(float(np.expm1(pred_log)), 1)
    pred_lower     = round(float(np.expm1(m["lower_bound"].predict(s2_row)[0])), 1)
    pred_upper     = round(float(np.expm1(m["upper_bound"].predict(s2_row)[0])), 1)
    clf_id         = m["classifier"].predict(s2_row)[0]
    clf_verdict    = m["label_encoder"].inverse_transform([clf_id])[0]

    profit        = pred_worldwide - float(budget)
    profit_pct    = (profit / float(budget)) * 100
    pred_verdict  = verdict_from_profit(profit_pct)
    agree         = (clf_verdict == pred_verdict)
    vs            = VERDICT_STYLES.get(pred_verdict, VERDICT_STYLES["AVERAGE"])

    conf_icon  = "🟢" if agree else "🟡"
    conf_text  = "High confidence — all models agree" if agree else f"Medium confidence — classifier suggested {clf_verdict}"
    roi_arrow  = "▲" if profit >= 0 else "▼"
    roi_color  = "#00E676" if profit >= 0 else "#FF5252"

    interval_pct = meta.get("interval_coverage", 0.70)
    range_span   = max(pred_upper - pred_lower, 1)
    center_pct   = min(max(int(((pred_worldwide - pred_lower) / range_span) * 100), 5), 95)

    st.markdown('<div class="result-wrap">', unsafe_allow_html=True)

    # ── Main result card ──────────────────────────────────────
    st.markdown(f"""
    <div class="result-card" style="border: 1px solid {vs['color']}22;">
        <div class="result-card-bg" style="background: radial-gradient(ellipse at 50% 0%, {vs['bg']}, transparent 70%);"></div>
        <div class="result-card-inner">
            <div class="result-eyebrow" style="color:{vs['color']}">Predicted Lifetime Worldwide</div>
            <div class="result-amount">₹{pred_worldwide:,.0f} Cr</div>
            <div class="result-range">Range: ₹{pred_lower:,.0f} – ₹{pred_upper:,.0f} Cr</div>
            <div class="result-roi" style="color:{roi_color}">
                {roi_arrow} ₹{abs(round(profit,1)):,.0f} Cr &nbsp;({round(profit_pct,1)}% ROI)
            </div>
            <div>
                <span class="verdict-badge" style="color:{vs['color']}; border-color:{vs['color']}44; background:{vs['bg']}; box-shadow: 0 0 20px {vs['glow']};">
                    {vs['emoji']} &nbsp;{pred_verdict}
                </span>
            </div>
            <div class="result-meta">{genre} &nbsp;·&nbsp; {language} &nbsp;·&nbsp; {'Franchise' if is_franchise else 'Original'}</div>
            <div class="confidence-text" style="color:{'#00E676' if agree else '#EF9F27'}; opacity:0.75;">
                {conf_icon} {conf_text}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence range bar ──────────────────────────────────
    st.markdown(f"""
    <div class="range-section">
        <div class="range-header">
            <span class="range-title">Confidence Range</span>
            <span class="range-coverage">{int(interval_pct*100)}% of films fall inside</span>
        </div>
        <div class="range-track">
            <div class="range-fill" style="left:0; width:100%; background:linear-gradient(90deg,
                rgba(127,119,221,0.1), {vs['color']}33, rgba(127,119,221,0.1));"></div>
            <div class="range-dot" style="left:{center_pct}%; background:{vs['color']};
                box-shadow: 0 0 16px {vs['glow']};"></div>
        </div>
        <div class="range-minmax">
            <span>₹{pred_lower:,.0f} Cr (low)</span>
            <span style="color:{vs['color']};">₹{pred_worldwide:,.0f} Cr</span>
            <span>₹{pred_upper:,.0f} Cr (high)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics ───────────────────────────────────────────────
    delta_color = "#00E676" if profit >= 0 else "#FF5252"
    delta_text  = f"+₹{round(profit,1)} Cr" if profit >= 0 else f"-₹{round(abs(profit),1)} Cr"
    st.markdown(f"""
    <div class="metrics-grid">
        <div class="metric-box">
            <div class="metric-lbl">Budget</div>
            <div class="metric-val">₹{budget:.0f}</div>
            <div class="metric-sub" style="color:rgba(255,255,255,0.2);">crore</div>
        </div>
        <div class="metric-box">
            <div class="metric-lbl">Opening Day</div>
            <div class="metric-val">₹{opening_day:.0f}</div>
            <div class="metric-sub" style="color:rgba(255,255,255,0.2);">crore</div>
        </div>
        <div class="metric-box">
            <div class="metric-lbl">Worldwide</div>
            <div class="metric-val">₹{pred_worldwide:.0f}</div>
            <div class="metric-sub" style="color:{delta_color};">{delta_text}</div>
        </div>
        <div class="metric-box">
            <div class="metric-lbl">ROI</div>
            <div class="metric-val" style="color:{delta_color};">{round(profit_pct,1)}%</div>
            <div class="metric-sub" style="color:rgba(255,255,255,0.2);">return</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Two-stage breakdown ───────────────────────────────────
    st.markdown(f"""
    <div class="stage-row">
        <div class="stage-box">
            <div class="stage-num">Stage 1 — Pre-release</div>
            <div class="stage-title">Opening Week Prediction</div>
            <div class="stage-val">₹{pred_opening_wk} Cr</div>
            <div class="stage-desc">Budget · Screens · Star · Genre</div>
        </div>
        <div class="stage-box" style="border-left-color: rgba(216,90,48,0.5); background: rgba(216,90,48,0.03);">
            <div class="stage-num" style="color:#D85A30;">Stage 2 — Lifetime</div>
            <div class="stage-title">Worldwide Collection</div>
            <div class="stage-val">₹{pred_worldwide} Cr</div>
            <div class="stage-desc">Opening week + all features</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Model votes ───────────────────────────────────────────
    reg_votes_html = ""
    clf_votes_html = ""
    try:
        for bname, bmodel in m["regressor"].estimators_:
            bpred = round(float(np.expm1(bmodel.predict(s2_row.values)[0])), 1)
            reg_votes_html += f'<div class="vote-item"><span>{bname.replace("_"," ").title()}</span><span class="vote-val">₹{bpred} Cr</span></div>'
    except Exception: pass
    reg_votes_html += f'<div class="vote-item final"><span><b>Ensemble</b></span><span class="vote-val">₹{pred_worldwide} Cr</span></div>'

    try:
        for bname, bmodel in m["classifier"].estimators_:
            bvote = m["label_encoder"].inverse_transform([bmodel.predict(s2_row.values)[0]])[0]
            bvs   = VERDICT_STYLES.get(bvote, VERDICT_STYLES["AVERAGE"])
            color = bvs["color"]
            emoji = bvs["emoji"]
            clf_votes_html += f'<div class="vote-item"><span>{bname.replace("_"," ").title()}</span><span class="vote-verdict" style="color:{color}">{emoji} {bvote}</span></div>'
    except Exception: pass
    final_vs = VERDICT_STYLES.get(clf_verdict, VERDICT_STYLES["AVERAGE"])
    fcolor = final_vs["color"]
    femoji = final_vs["emoji"]
    clf_votes_html += f'<div class="vote-item final"><span><b>Ensemble</b></span><span class="vote-verdict" style="color:{fcolor}">{femoji} {clf_verdict}</span></div>'

    st.markdown(f"""
    <div class="votes-grid">
        <div>
            <div class="votes-col-title">Regressor Votes (₹ Cr)</div>
            {reg_votes_html}
        </div>
        <div>
            <div class="votes-col-title">Classifier Votes (Verdict)</div>
            {clf_votes_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Verdict guide ─────────────────────────────────────────
    vg_items = ""
    for vname, vstyle in VERDICT_STYLES.items():
        vg_items += f"""
        <div class="vg-item" style="border-color:{vstyle['color']}22; background:{vstyle['bg']};">
            <div class="vg-emoji">{vstyle['emoji']}</div>
            <div class="vg-name" style="color:{vstyle['color']};">{vname}</div>
            <div class="vg-pct">{vstyle['desc']}</div>
        </div>"""
    st.markdown(f'<div class="vg-wrap">{vg_items}</div>', unsafe_allow_html=True)

    # ── How it was made ───────────────────────────────────────
    with st.expander("ℹ️ How this prediction was made"):
        st.markdown(f"""
**Star:** {star_note}  
**Director:** {dir_note}  
**Ensemble:** {' · '.join(top3_reg)} → Ridge meta-learner  
**Features used:** Budget, Opening Day, Screens, Star Power, Director Power, Genre, Language, Season, Franchise, Log Budget, Overseas Ratio, Festival Release  
**Two-stage:** Stage 1 predicts opening week → Stage 2 predicts lifetime  
**Confidence interval:** Quantile regression at 10th/90th percentile · {int(interval_pct*100)}% coverage on test set
        """)

    st.markdown('</div>', unsafe_allow_html=True)

    save_prediction({
        "movie_name":movie_name,"star":star_name,"director":director_name,
        "language":language,"genre":genre,"budget":budget,"opening_day":opening_day,
        "screens":screens,"is_franchise":is_franchise,"pred_worldwide":pred_worldwide,
        "pred_lower":pred_lower,"pred_upper":pred_upper,"pred_opening_wk":pred_opening_wk,
        "pred_profit":profit,"pred_profit_pct":profit_pct,
        "pred_verdict":pred_verdict,"clf_verdict":clf_verdict,
        "confidence":conf_text,"ensemble_agreement":agree,
        "star_power_used":star_val,"dir_power_used":dir_val,
    })

# ── Footer ────────────────────────────────────────────────────
st.markdown(f"""
<div class="cp-footer">
    CinePredict &nbsp;·&nbsp; 12-Algorithm Stacking Ensemble &nbsp;·&nbsp;
    Two-Stage Prediction &nbsp;·&nbsp; 1150+ Indian Films 2017–2024<br>
    R² {meta.get('reg_r2',0):.3f} &nbsp;·&nbsp;
    MAE ₹{meta.get('reg_mae_cr',0):.0f}Cr &nbsp;·&nbsp;
    Accuracy {meta.get('clf_accuracy',0)*100:.0f}% &nbsp;·&nbsp;
    For educational purposes only
</div>
""", unsafe_allow_html=True)
