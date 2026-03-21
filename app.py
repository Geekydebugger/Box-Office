# ============================================================
#   INDIAN BOX OFFICE PREDICTOR — app.py  (v4)
#   ✦ Rating input
#   ✦ Confidence interval (range display)
#   ✦ Two-stage prediction (opening week → lifetime)
#   ✦ 12-algo leaderboard + base model votes
#   ✦ SQLite history
#   Run with:  streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, sqlite3, datetime
from difflib import get_close_matches

# ── Constants ─────────────────────────────────────────────────
DB_PATH    = "models/box_office.db"
GENRE_LIST = ["Action","Adventure","Biography","Comedy","Crime","Drama",
              "Family","Fantasy","Historical","Horror","Musical","Mystery",
              "Romance","Sci-Fi","Sports","Supernatural","Thriller"]

# ── Helpers ───────────────────────────────────────────────────
def verdict_from_profit(p):
    if p < 0:    return "FLOP"
    elif p < 50: return "AVERAGE"
    elif p < 100: return "HIT"
    elif p < 200: return "SUPER HIT"
    else:         return "BLOCKBUSTER"

def get_season(month):
    if month in [1,10,11,12]: return "Holiday"
    elif month in [3,4,5]:    return "Summer"
    elif month in [6,7]:      return "Monsoon"
    else:                      return "Normal"

def verdict_color(v):
    return {
        "BLOCKBUSTER": ("#FFD700","#1a1a00"),
        "SUPER HIT":   ("#4CAF50","#001a00"),
        "HIT":         ("#2196F3","#001020"),
        "AVERAGE":     ("#9E9E9E","#111111"),
        "FLOP":        ("#FF5252","#1a0000"),
    }.get(v, ("#9E9E9E","#111111"))

def verdict_emoji(v):
    return {
        "BLOCKBUSTER": "🔥",
        "SUPER HIT":   "⭐",
        "HIT":         "✅",
        "AVERAGE":     "➡️",
        "FLOP":        "📉",
    }.get(v,"🎬")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="Box Office Predictor", page_icon="🎬", layout="centered")

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 700; text-align: center;
        background: linear-gradient(135deg, #7F77DD, #D85A30);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle { text-align:center; color:#888; font-size:0.95rem; margin-bottom:1.2rem; }
    .result-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px; padding: 2rem; text-align: center;
        margin-top: 1.5rem; border: 1px solid #333;
    }
    .result-crore  { font-size: 3rem; font-weight: 800; color: #FFD700; }
    .result-range  { font-size: 1rem; color: #aaa; margin-top: 0.2rem; }
    .result-label  { font-size: 0.8rem; color: #777; text-transform: uppercase; letter-spacing: 0.1em; }
    .verdict-badge {
        display: inline-block; padding: 0.4rem 1.4rem;
        border-radius: 30px; font-size: 1.1rem; font-weight: 700; margin-top: 0.8rem;
    }
    .stage-box {
        background: rgba(127,119,221,0.08); border: 1px solid rgba(127,119,221,0.25);
        border-radius: 10px; padding: 0.8rem 1rem; margin: 0.5rem 0; font-size: 0.9rem;
    }
    .vote-row {
        display: flex; justify-content: space-between; padding: 0.3rem 0.8rem;
        border-radius: 8px; background: rgba(255,255,255,0.04); margin: 3px 0; font-size: 0.88rem;
    }
    .algo-badge {
        display:inline-block; background:#1e2a3a; color:#90caf9;
        border-radius:6px; padding:0.15rem 0.6rem; font-size:0.73rem; font-weight:600; margin:2px;
    }
    .stAlert { border-radius: 10px; }
</style>
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

# ── DB helpers ────────────────────────────────────────────────
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
            p["budget"],p["opening_day"],p["screens"],p["release_month"],p["release_year"],
            int(p["is_franchise"]),
            p["pred_worldwide"],p["pred_lower"],p["pred_upper"],p["pred_opening_wk"],
            p["pred_profit"],p["pred_profit_pct"],
            p["pred_verdict"],p["clf_verdict"],p["confidence"],
            int(p["ensemble_agreement"]),p["star_power_used"],p["dir_power_used"]
        ))
        con.commit()
    except Exception:
        pass
    con.close()

def load_history():
    if not os.path.exists(DB_PATH): return pd.DataFrame()
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("""SELECT movie_name,genre,pred_worldwide,pred_lower,pred_upper,
            pred_opening_wk,budget,pred_profit_pct,pred_verdict,clf_verdict,
            star,director,predicted_at
            FROM predictions ORDER BY id DESC LIMIT 100""", con)
    except Exception:
        df = pd.DataFrame()
    con.close()
    return df

def load_training_history():
    if not os.path.exists(DB_PATH): return pd.DataFrame()
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("""SELECT run_at,n_movies,reg_r2,reg_mae_cr,reg_cv_r2,
            clf_accuracy,best_reg_algo,best_clf_algo,ensemble_type
            FROM training_runs ORDER BY id DESC LIMIT 10""", con)
    except Exception:
        df = pd.DataFrame()
    con.close()
    return df

# ── Header ────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎬 Box Office Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">5-Class Verdict · Two-Stage · Confidence Intervals · 12-Algorithm Ensemble · 1200+ Films</div>',
            unsafe_allow_html=True)

models, missing = load_models()
if models is None:
    st.error(f"⚠️ Models not found. Run `python main.py` first.\n\nMissing: {missing}")
    st.stop()

m        = models
meta     = m["meta"]
star_map = m["star_power_map"]
dir_map  = m["director_power_map"]
lang_enc = m["label_language"]
sea_enc  = m["label_season"]
gen_enc  = m["label_genre"]
top3_reg = meta.get("top3_reg", ["XGBoost","RandomForest","GradientBoosting"])
top3_clf = meta.get("top3_clf", ["XGBoost","RandomForest","GradientBoosting"])
known_stars     = sorted(star_map.index.tolist())
known_directors = sorted(dir_map.index.tolist())

# Ensemble badge
st.markdown(
    '<div style="text-align:center; margin-bottom:1rem;">'
    + "".join(f'<span class="algo-badge">⚡{n}</span>' for n in top3_reg)
    + ' <span style="color:#888;font-size:0.78rem;">→ Ridge meta · Two-Stage</span>'
    + '</div>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📋 History", "📈 Model Stats"])

# ══════════════════════════════════════════════════════════════
#  TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 🎥 Movie Details")

    col1, col2 = st.columns(2)
    with col1:
        movie_name  = st.text_input("Movie Name", placeholder="e.g. Tiger 4")
        star_name   = st.text_input("Lead Star", placeholder="e.g. Salman Khan",
                                    help=f"Known: {', '.join(known_stars[:6])}...")
        budget      = st.number_input("Budget (₹ Crore)", min_value=1.0, max_value=2000.0, value=100.0, step=5.0)
        opening_day = st.number_input("Expected Opening Day (₹ Crore)", min_value=0.1, max_value=500.0, value=20.0, step=1.0)
        genre       = st.selectbox("Genre", options=GENRE_LIST, index=GENRE_LIST.index("Action"))

    with col2:
        director_name = st.text_input("Director", placeholder="e.g. Rohit Shetty",
                                      help=f"Known: {', '.join(known_directors[:6])}...")
        language = st.selectbox("Language", options=sorted(lang_enc.classes_.tolist()),
                                index=list(lang_enc.classes_).index("Hindi")
                                if "Hindi" in lang_enc.classes_ else 0)
        screens  = st.number_input("Worldwide Screens", min_value=100, max_value=15000, value=4000, step=100)
        release_month = st.selectbox("Release Month", options=list(range(1,13)),
                                     format_func=lambda x: ["","Jan","Feb","Mar","Apr","May","Jun",
                                                             "Jul","Aug","Sep","Oct","Nov","Dec"][x],
                                     index=9)
        release_year = st.selectbox("Release Year", [2024,2025,2026], index=1)

    col3, col4 = st.columns(2)
    with col3:
        is_franchise = st.checkbox("🔁 Sequel / Franchise film?")
    with col4:
        rating = st.slider("Expected Rating (IMDb/audience)", min_value=1.0, max_value=10.0,
                           value=6.5, step=0.1,
                           help="Use expected critic/audience score — affects lifetime legs")

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Box Office", use_container_width=True, type="primary")

    if predict_btn:
        if not movie_name.strip():
            st.warning("Please enter a movie name.")
            st.stop()

        # Resolve star & director
        star_match = get_close_matches(star_name.strip().title(), star_map.index.tolist(), n=1, cutoff=0.6)
        star_val   = float(star_map[star_match[0]]) if star_match else float(meta["global_star_mean"])
        star_note  = f"Matched: **{star_match[0]}**" if star_match else "Using average star power"

        dir_match  = get_close_matches(director_name.strip().title(), dir_map.index.tolist(), n=1, cutoff=0.6)
        dir_val    = float(dir_map[dir_match[0]]) if dir_match else float(meta["global_director_mean"])
        dir_note   = f"Matched: **{dir_match[0]}**" if dir_match else "Using average director power"

        lang_val   = int(lang_enc.transform([language])[0]) if language in lang_enc.classes_ else 0
        season_str = get_season(release_month)
        season_val = int(sea_enc.transform([season_str])[0]) if season_str in sea_enc.classes_ else 0
        genre_val  = int(gen_enc.transform([genre])[0]) if genre in gen_enc.classes_ else 0
        fest_val   = 1 if release_month in [4,5,10,11,12] else 0
        log_bud    = np.log1p(budget)
        bud_sq     = float(budget) ** 2

        # ── Stage 1: predict opening week ────────────────────
        s1_row = pd.DataFrame([{
            "Budget":budget,"Screens":screens,"Language_Label":lang_val,
            "Season_Label":season_val,"Franchise":int(is_franchise),
            "Screens_to_Budget":screens/budget,"Release_Year":release_year,
            "Genre_Label":genre_val,"Log_Budget":log_bud,"Budget_Squared":bud_sq,
            "Rating":rating,"Festival_Release":fest_val,
            "Star_Power":star_val,"Director_Power":dir_val,
        }])
        pred_opening_wk = round(float(np.expm1(m["stage1_model"].predict(s1_row)[0])), 1)

        # ── Stage 2: predict lifetime ─────────────────────────
        s2_row = pd.DataFrame([{
            "Budget":float(budget),"Opening_Day":float(opening_day),"Screens":float(screens),
            "Language_Label":lang_val,"Season_Label":season_val,"Franchise":int(is_franchise),
            "Opening_to_Budget":float(opening_day)/float(budget),
            "Screens_to_Budget":float(screens)/float(budget),
            "Opening_per_Screen":float(opening_day)/float(screens) if screens>0 else 0,
            "Release_Year":int(release_year),"Genre_Label":genre_val,
            "Rating":float(rating),"Log_Budget":log_bud,"Budget_Squared":bud_sq,
            "Overseas_Ratio":0.15,"Rating_x_Budget":float(rating)*float(budget),
            "Festival_Release":fest_val,
            "Star_Power":star_val,"Director_Power":dir_val,
            "Pred_Opening_Week":pred_opening_wk,
            "Log_Opening_Week":np.log1p(pred_opening_wk),
        }])

        pred_log       = m["regressor"].predict(s2_row)[0]
        pred_worldwide = round(float(np.expm1(pred_log)), 1)
        pred_lower     = round(float(np.expm1(m["lower_bound"].predict(s2_row)[0])), 1)
        pred_upper     = round(float(np.expm1(m["upper_bound"].predict(s2_row)[0])), 1)
        clf_id         = m["classifier"].predict(s2_row)[0]
        clf_verdict    = m["label_encoder"].inverse_transform([clf_id])[0]

        profit       = pred_worldwide - float(budget)
        profit_pct   = (profit / float(budget)) * 100
        pred_verdict = verdict_from_profit(profit_pct)
        agree        = (clf_verdict == pred_verdict)

        if agree:
            conf_label = "🟢 High confidence — all models agree"
            conf_color = "#1D9E75"
        else:
            conf_label = f"🟡 Medium confidence — classifier suggested {clf_verdict}"
            conf_color = "#EF9F27"

        roi_label   = f"{'▲' if profit>=0 else '▼'} ₹{abs(round(profit,1))} Cr  ({round(profit_pct,1)}%)"
        v_color, v_bg = verdict_color(pred_verdict)
        emoji         = verdict_emoji(pred_verdict)
        interval_pct  = meta.get("interval_coverage", 0.7)

        # ── Result card ───────────────────────────────────────
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Predicted Lifetime Worldwide Collection</div>
            <div class="result-crore">₹{pred_worldwide:,.1f} Cr</div>
            <div class="result-range">
                📊 Range: ₹{pred_lower:,.0f} Cr — ₹{pred_upper:,.0f} Cr
                <span style="color:#666; font-size:0.8rem;"> ({interval_pct*100:.0f}% confidence)</span>
            </div>
            <div style="color:#ccc; margin-top:0.3rem; font-size:0.95rem;">{roi_label}</div>
            <div class="verdict-badge" style="background:{v_color}; color:{v_bg}; margin-top:1rem;">
                {emoji} {pred_verdict}
            </div>
            <div style="color:#666; font-size:0.8rem; margin-top:0.8rem;">
                Genre: {genre} · Rating: {rating}/10 · Season: {season_str} · Franchise: {'Yes' if is_franchise else 'No'}
            </div>
            <div style="color:{conf_color}; font-size:0.85rem; margin-top:0.5rem;">{conf_label}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Two-stage breakdown ───────────────────────────────
        # ── Verdict guide ─────────────────────────────────────
        st.markdown("#### 🎯 Verdict Guide")
        vcols = st.columns(5)
        verdicts = [
            ("📉","FLOP",       "#FF5252","Loss making"),
            ("➡️","AVERAGE",   "#9E9E9E","Break even"),
            ("✅","HIT",        "#2196F3","50–100% profit"),
            ("⭐","SUPER HIT",  "#4CAF50","100–200% profit"),
            ("🔥","BLOCKBUSTER","#FFD700","200%+ profit"),
        ]
        for col,(emoji,label,color,desc) in zip(vcols, verdicts):
            col.markdown(f"""
            <div style="text-align:center; padding:0.5rem;
                        border-radius:8px; border:1px solid {color}55;">
                <div style="font-size:1.2rem;">{emoji}</div>
                <div style="color:{color}; font-weight:700; font-size:0.78rem;">{label}</div>
                <div style="color:#888; font-size:0.68rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 🎯 Two-Stage Breakdown")
        st.markdown(f"""
        <div class="stage-box">
            <b>Stage 1 — Opening Week Prediction</b><br>
            Pre-release features (budget, screens, star, genre, rating, festival) →
            <span style="color:#FFD700;font-weight:600;"> ₹{pred_opening_wk} Cr</span> predicted week-1 collection
        </div>
        <div class="stage-box">
            <b>Stage 2 — Lifetime Prediction</b><br>
            Stage-1 result + opening day estimate + all features →
            <span style="color:#FFD700;font-weight:600;"> ₹{pred_worldwide} Cr</span> lifetime worldwide
        </div>
        """, unsafe_allow_html=True)

        # ── Metrics ───────────────────────────────────────────
        st.markdown("#### 📊 Breakdown")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Budget",       f"₹{budget} Cr")
        m2.metric("Opening Day",  f"₹{opening_day} Cr")
        m3.metric("Opening Week", f"₹{pred_opening_wk} Cr")
        m4.metric("Worldwide",    f"₹{pred_worldwide} Cr",
                  delta=f"₹{round(profit,1)} Cr" if profit>=0 else f"-₹{round(abs(profit),1)} Cr")
        m5.metric("ROI",          f"{round(profit_pct,1)}%")

        # ── Confidence range bar ──────────────────────────────
        st.markdown("#### 📏 Confidence Range")
        range_span = max(pred_upper - pred_lower, 1)
        center_pct = int(((pred_worldwide - pred_lower) / range_span) * 100)
        st.markdown(f"""
        <div style="margin:0.5rem 0 0.2rem; font-size:0.85rem; color:var(--color-text-secondary);">
            ₹{pred_lower} Cr &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            <b>₹{pred_worldwide} Cr</b>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ₹{pred_upper} Cr
        </div>
        <div style="background:rgba(127,119,221,0.15); border-radius:8px; height:18px; position:relative; overflow:hidden;">
            <div style="position:absolute; left:0; top:0; height:100%; width:100%;
                        background:linear-gradient(90deg,rgba(127,119,221,0.2),rgba(127,119,221,0.5),rgba(127,119,221,0.2));
                        border-radius:8px;"></div>
            <div style="position:absolute; left:{center_pct}%; top:0; height:100%; width:3px;
                        background:#FFD700; border-radius:2px;"></div>
        </div>
        <div style="font-size:0.75rem; color:#666; margin-top:0.2rem;">
            Yellow line = point estimate · Shaded = {interval_pct*100:.0f}% confidence interval
        </div>
        """, unsafe_allow_html=True)

        # ── Model votes ───────────────────────────────────────
        st.markdown("#### 🗳️ Model Vote Breakdown")
        vc1, vc2 = st.columns(2)
        with vc1:
            st.markdown("**Regressor votes**")
            try:
                for bname, bmodel in m["regressor"].estimators_:
                    bpred = round(float(np.expm1(bmodel.predict(s2_row.values)[0])), 1)
                    st.markdown(f'<div class="vote-row"><span>{bname.replace("_"," ").title()}</span>'
                                f'<span style="color:#FFD700;">₹{bpred} Cr</span></div>',
                                unsafe_allow_html=True)
            except Exception:
                st.caption("(votes unavailable)")
            st.markdown(f'<div class="vote-row" style="border:1px solid #7F77DD;">'
                        f'<span><b>Ensemble</b></span>'
                        f'<span style="color:#FFD700;"><b>₹{pred_worldwide} Cr</b></span></div>',
                        unsafe_allow_html=True)

        with vc2:
            st.markdown("**Classifier votes**")
            try:
                for bname, bmodel in m["classifier"].estimators_:
                    bvote = m["label_encoder"].inverse_transform([bmodel.predict(s2_row.values)[0]])[0]
                    e = verdict_emoji(bvote)
                    st.markdown(f'<div class="vote-row"><span>{bname.replace("_"," ").title()}</span>'
                                f'<span>{e} {bvote}</span></div>', unsafe_allow_html=True)
            except Exception:
                st.caption("(votes unavailable)")
            e2 = verdict_emoji(clf_verdict)
            st.markdown(f'<div class="vote-row" style="border:1px solid #7F77DD;">'
                        f'<span><b>Ensemble</b></span>'
                        f'<span><b>{e2} {clf_verdict}</b></span></div>',
                        unsafe_allow_html=True)

        # ── Expander ──────────────────────────────────────────
        with st.expander("ℹ️ How this prediction was made"):
            st.markdown(f"**Star:** {star_note}")
            st.markdown(f"**Director:** {dir_note}")
            st.markdown(f"**Season:** {season_str} · **Genre:** {genre} · **Rating:** {rating}/10")
            st.markdown(f"**Festival release:** {'Yes' if fest_val else 'No'}")
            st.markdown(f"**Top 3 Regressors:** {', '.join(top3_reg)}")
            st.markdown(f"**Top 3 Classifiers:** {', '.join(top3_clf)}")
            st.markdown(f"**Interval coverage:** {interval_pct*100:.0f}% of test films fell inside the predicted range")
            st.markdown("""
---
**New in v4:**
- **Rating** used as a feature — higher rated films have longer theatrical legs
- **Log_Budget + Budget²** — captures non-linear blockbuster scale effect
- **Overseas_Ratio** — separates pan-India films from regional ones
- **Festival_Release** — Eid/Diwali/Christmas releases modelled separately
- **Two-stage model** — Stage 1 predicts opening week, Stage 2 uses it to predict lifetime
- **Confidence interval** — quantile regression gives a realistic range, not fake precision
- **5-class verdict** — FLOP · AVERAGE · HIT · SUPER HIT · BLOCKBUSTER (more accurate than 7 classes)
            """)

        save_prediction({
            "movie_name":movie_name,"star":star_name,"director":director_name,
            "language":language,"genre":genre,"budget":budget,"opening_day":opening_day,
            "screens":screens,"release_month":release_month,"release_year":release_year,
            "is_franchise":is_franchise,"pred_worldwide":pred_worldwide,
            "pred_lower":pred_lower,"pred_upper":pred_upper,"pred_opening_wk":pred_opening_wk,
            "pred_profit":profit,"pred_profit_pct":profit_pct,
            "pred_verdict":pred_verdict,"clf_verdict":clf_verdict,
            "confidence":conf_label,"ensemble_agreement":agree,
            "star_power_used":star_val,"dir_power_used":dir_val,
        })

# ══════════════════════════════════════════════════════════════
#  TAB 2 — HISTORY
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📋 Prediction History")
    df_hist = load_history()
    if df_hist.empty:
        st.info("No predictions yet. Make one in the Predict tab!")
    else:
        st.markdown(f"**{len(df_hist)} prediction(s) stored**")
        df_hist["Range"] = df_hist.apply(
            lambda r: f"₹{r['pred_lower']:.0f}–₹{r['pred_upper']:.0f} Cr"
                      if pd.notna(r.get("pred_lower")) else "—", axis=1)
        df_display = df_hist.rename(columns={
            "movie_name":"Movie","genre":"Genre",
            "pred_worldwide":"Predicted (Cr)","pred_opening_wk":"Opening Wk (Cr)",
            "budget":"Budget (Cr)","pred_profit_pct":"ROI %",
            "pred_verdict":"Verdict","star":"Star","predicted_at":"When",
        })
        df_display["ROI %"]          = df_display["ROI %"].round(1)
        df_display["Predicted (Cr)"] = df_display["Predicted (Cr)"].round(1)
        df_display["When"]           = pd.to_datetime(df_display["When"]).dt.strftime("%d %b %Y %H:%M")
        st.dataframe(df_display[["Movie","Genre","Predicted (Cr)","Range","Opening Wk (Cr)",
                                  "Budget (Cr)","ROI %","Verdict","Star","When"]],
                     use_container_width=True, hide_index=True)
        if len(df_hist) >= 3:
            st.bar_chart(df_hist["pred_verdict"].value_counts())

# ══════════════════════════════════════════════════════════════
#  TAB 3 — MODEL STATS
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Model Performance")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("R² Score",    f"{meta.get('reg_r2',0):.3f}")
    c2.metric("MAE",         f"₹{meta.get('reg_mae_cr',0):.0f} Cr")
    c3.metric("Clf Accuracy",f"{meta.get('clf_accuracy',0)*100:.1f}%")
    c4.metric("CI Coverage", f"{meta.get('interval_coverage',0)*100:.0f}%")
    c5.metric("CI Width",    f"₹{meta.get('interval_avg_width',0):.0f} Cr")

    st.markdown(f"**Selected Regressors:** {' · '.join(top3_reg)}")
    st.markdown(f"**Selected Classifiers:** {' · '.join(top3_clf)}")

    st.markdown("**What's new in v4:**")
    st.markdown("""
    - ✅ **Rating** — audience/critic score now used as a feature
    - ✅ **Log_Budget + Budget²** — non-linear budget scale
    - ✅ **Overseas_Ratio** — pan-India vs regional film signal
    - ✅ **Festival_Release** — Eid/Diwali/Christmas flag
    - ✅ **Two-stage prediction** — opening week first, then lifetime
    - ✅ **Confidence interval** — realistic range via quantile regression
    """)

    reg_scores = meta.get("reg_scores", {})
    clf_scores = meta.get("clf_scores", {})

    if reg_scores:
        st.markdown("#### 🏆 Regressor Leaderboard")
        reg_df = pd.DataFrame([
            {"Algorithm":k, "Test R²":round(v["r2"],4),
             "MAE (Cr)":round(v["mae_cr"],1), "CV R²":round(v["cv_r2"],4),
             "In Ensemble":"★" if k in top3_reg else ""}
            for k,v in sorted(reg_scores.items(),key=lambda x:x[1]["cv_r2"],reverse=True)
        ])
        st.dataframe(reg_df, use_container_width=True, hide_index=True)
        st.bar_chart(pd.DataFrame({"CV R²":[reg_scores[k]["cv_r2"] for k in reg_scores]},
                     index=list(reg_scores.keys())).sort_values("CV R²",ascending=False))

    if clf_scores:
        st.markdown("#### 🏆 Classifier Leaderboard")
        clf_df = pd.DataFrame([
            {"Algorithm":k, "Test Accuracy":round(v["acc"],4),
             "CV Accuracy":round(v["cv_acc"],4),
             "In Ensemble":"★" if k in top3_clf else ""}
            for k,v in sorted(clf_scores.items(),key=lambda x:x[1]["cv_acc"],reverse=True)
        ])
        st.dataframe(clf_df, use_container_width=True, hide_index=True)
        st.bar_chart(pd.DataFrame({"CV Accuracy":[clf_scores[k]["cv_acc"] for k in clf_scores]},
                     index=list(clf_scores.keys())).sort_values("CV Accuracy",ascending=False))

    st.markdown("#### 🗂️ Training Run History")
    df_tr = load_training_history()
    if df_tr.empty:
        st.info("No training history. Run `python main.py`.")
    else:
        df_tr["run_at"]       = pd.to_datetime(df_tr["run_at"]).dt.strftime("%d %b %Y %H:%M")
        df_tr["clf_accuracy"] = (df_tr["clf_accuracy"]*100).round(1).astype(str)+"%"
        df_tr["reg_r2"]       = df_tr["reg_r2"].round(4)
        st.dataframe(df_tr.rename(columns={
            "run_at":"When","n_movies":"Movies","reg_r2":"R²","reg_mae_cr":"MAE Cr",
            "reg_cv_r2":"CV R²","clf_accuracy":"Accuracy",
            "best_reg_algo":"Best Reg","best_clf_algo":"Best Clf","ensemble_type":"Ensemble"
        }), use_container_width=True, hide_index=True)

    if os.path.exists("models/results.png"):
        st.markdown("#### 📊 Latest Training Chart")
        st.image("models/results.png", use_container_width=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666; font-size:0.8rem;'>"
    "v4 · 5-Class Verdict · Two-Stage · Confidence Intervals · 12-Algorithm Ensemble · Indian Box Office 2017–2024"
    "</div>", unsafe_allow_html=True)
