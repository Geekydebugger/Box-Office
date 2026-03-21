# 🎬 CinePredict — Indian Box Office Predictor

> Predict lifetime worldwide collection and verdict for any Indian film using a 12-algorithm stacking ensemble with two-stage prediction and confidence intervals.

**R² = 0.817 · MAE = ₹17.9 Cr · Verdict Accuracy = 71.3% · 1150+ films · 2017–2024**

---

## 📁 Project Structure

```
boxxoffice/
│
├── data/                               ← All CSV files go here
│   ├── Indian_Movies_2017.csv
│   ├── Indian_Movies_2018.csv
│   ├── Indian_Movies_2019.csv
│   ├── Indian_Movies_2020_updated.csv
│   ├── Indian_Movies_2021_updated.csv
│   ├── Indian_Movies_2022_updated.csv
│   ├── Indian_Movies_2023_updated.csv
│   └── Indian_Movies_2024_updated.csv
│
├── models/                             ← Auto-created after training
│   ├── regressor.pkl                   ← Stacking ensemble regressor
│   ├── classifier.pkl                  ← Stacking ensemble classifier
│   ├── stage1_model.pkl                ← Opening week predictor
│   ├── lower_bound.pkl                 ← Confidence interval lower
│   ├── upper_bound.pkl                 ← Confidence interval upper
│   ├── label_encoder.pkl
│   ├── label_language.pkl
│   ├── label_season.pkl
│   ├── label_genre.pkl
│   ├── star_power_map.pkl
│   ├── director_power_map.pkl
│   ├── scaler_s2.pkl
│   ├── meta.pkl
│   ├── box_office.db                   ← SQLite database
│   └── results.png                     ← Training charts
│
├── main.py                             ← Train all models
├── app.py                              ← Streamlit web app
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place CSV files
Put all 8 CSV files inside the `data/` folder.

### 3. Train models
```bash
python main.py
```

This will:
- Load and clean all CSV files (1150+ movies)
- Engineer 19 features including Rating, Log_Budget, Overseas_Ratio, Festival_Release
- Benchmark 12 algorithms for both regressor and classifier
- Tune CatBoost and XGBoost with RandomizedSearchCV
- Auto-select top 3 algorithms → build stacking ensemble
- Train two-stage predictor (opening week → lifetime)
- Train confidence interval models (quantile regression)
- Save all models to `models/`
- Log training run to SQLite database
- Generate and save result charts

**Expected scores:**
- R² Score: 0.80+
- MAE: under ₹20 Cr
- Classifier Accuracy: 70%+
- Training time: 15–25 minutes

### 4. Run the web app
```bash
streamlit run app.py
```
Opens at http://localhost:8501

---

## 🧠 How It Works

### Features Used (19 total)

| Feature | Description |
|---|---|
| Budget | Film budget in ₹ Crore |
| Opening_Day | First day collection estimate |
| Screens | Worldwide screen count |
| Star_Power | Lead star's historical median collection |
| Director_Power | Director's historical average collection |
| Language_Label | Encoded language (Hindi/Tamil/Telugu etc.) |
| Genre_Label | Encoded genre (Action/Drama/Comedy etc.) |
| Season_Label | Release season (Holiday/Summer/Monsoon/Normal) |
| Franchise | Sequel/franchise flag |
| Rating | Expected IMDb/audience score |
| Log_Budget | Log-transformed budget (non-linear scale) |
| Budget_Squared | Budget² (blockbuster scale effect) |
| Overseas_Ratio | Overseas/Worldwide ratio |
| Rating_x_Budget | Quality × scale interaction |
| Festival_Release | Eid/Diwali/Christmas/Independence Day flag |
| Opening_to_Budget | Opening day / budget ratio |
| Screens_to_Budget | Screens / budget ratio |
| Opening_per_Screen | Opening day / screens |
| Release_Year | Year of release |

### Two-Stage Architecture
```
Stage 1 — Pre-release features only
(Budget, Screens, Star, Genre, Rating, Festival)
        ↓
Predicts Opening Week Collection

Stage 2 — All features + Stage 1 output
        ↓
Predicts Lifetime Worldwide Collection
```

### Stacking Ensemble
```
12 Algorithms benchmarked
        ↓
Top 3 auto-selected by combined CV R² + Test R²
        ↓
CatBoost + XGBoost + GradientBoosting
        ↓
Ridge Regression (meta-learner)
        ↓
Final Prediction
```

### Confidence Intervals
Two quantile regression models trained at 10th and 90th percentile give a realistic range instead of a single number.

### 5-Class Verdict System

| Verdict | Profit % |
|---|---|
| 📉 FLOP | Loss making (< 0%) |
| ➡️ AVERAGE | Break even (0–50%) |
| ✅ HIT | 50–100% profit |
| ⭐ SUPER HIT | 100–200% profit |
| 🔥 BLOCKBUSTER | 200%+ profit |

---

## 🌐 Deploy Free on Streamlit Cloud

1. Push this folder to a GitHub repo
2. Go to https://streamlit.io/cloud
3. Click **New app** → select your repo → set main file as `app.py`
4. Click **Deploy** — get a public shareable link!

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| R² Score | 0.817 |
| MAE | ₹17.9 Crore |
| CV R² | 0.8928 |
| Classifier Accuracy | 71.3% |
| Training data | 1150 films |
| Years covered | 2017–2024 |
| Languages | Hindi, Telugu, Tamil, Malayalam, Kannada, Bengali, Marathi, Punjabi + more |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| ML Models | XGBoost, LightGBM, CatBoost, RandomForest, GradientBoosting, Ridge, Lasso, ElasticNet, SVR, KNN, ExtraTrees, AdaBoost |
| Ensemble | Scikit-learn StackingRegressor / StackingClassifier |
| Tuning | RandomizedSearchCV (40 iterations, 3-fold CV) |
| Confidence | GradientBoostingRegressor (quantile loss) |
| Backend DB | SQLite via sqlite3 |
| Web App | Streamlit |
| Data | Pandas, NumPy |
| Charts | Matplotlib, Seaborn |

---

## ⚠️ Disclaimer

This model is for **educational purposes only**. Box office prediction has inherent uncertainty — even the best models cannot predict word-of-mouth, reviews, or competition. Do not use for investment decisions.

---

*CinePredict v6 — Built with Python & Streamlit*
