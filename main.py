# ================================================================
#   INDIAN BOX OFFICE PREDICTOR — main.py  (v5 + SMOTE)
#   ✦ Quick wins  : Rating, Log_Budget, Budget_Squared, Overseas_Ratio
#   ✦ Confidence  : Quantile regression → prediction range
#   ✦ Two-stage   : Stage-1 predicts opening weekend,
#                   Stage-2 uses it to predict lifetime
#   ✦ 12-algo benchmark + auto stacking ensemble
#   ✦ SMOTE class balancing for classifier training
#   ✦ SQLite logging
# ================================================================

import pandas as pd
import numpy as np
import glob, os, joblib, sqlite3, json, datetime, warnings
import matplotlib.pyplot as plt
import seaborn as sns
from difflib import get_close_matches
from collections import Counter
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ── Install imbalanced-learn if missing ──────────────────────
try:
    from imblearn.over_sampling import SMOTE, SVMSMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
    print("  ✓ imbalanced-learn detected")
except ImportError:
    import subprocess, sys
    print("  Installing imbalanced-learn...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "imbalanced-learn", "--quiet"])
    from imblearn.over_sampling import SMOTE, SVMSMOTE
    HAS_IMBLEARN = True
    print("  ✓ imbalanced-learn installed")

from sklearn.model_selection  import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.metrics          import mean_absolute_error, r2_score, accuracy_score, confusion_matrix
from sklearn.linear_model     import Ridge, Lasso, ElasticNet, BayesianRidge, LogisticRegression
from sklearn.tree             import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble         import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor,
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier,
    StackingRegressor, StackingClassifier,
)
from sklearn.svm              import SVR, SVC
from sklearn.neighbors        import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes      import GaussianNB
from xgboost                  import XGBRegressor, XGBClassifier
from scipy.stats              import randint, uniform
from sklearn.model_selection  import RandomizedSearchCV

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# ================================================================
#  DATABASE
# ================================================================
DB_PATH = "models/box_office.db"

def init_db():
    os.makedirs("models", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS training_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_at TEXT, n_movies INTEGER,
        reg_r2 REAL, reg_mae_cr REAL, reg_cv_r2 REAL,
        clf_accuracy REAL, best_reg_algo TEXT, best_clf_algo TEXT,
        ensemble_type TEXT, algo_scores TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        predicted_at TEXT, movie_name TEXT, star TEXT, director TEXT,
        language TEXT, genre TEXT, budget REAL, opening_day REAL,
        screens INTEGER, release_month INTEGER, release_year INTEGER,
        is_franchise INTEGER,
        pred_worldwide REAL, pred_lower REAL, pred_upper REAL,
        pred_opening_wk REAL,
        pred_profit REAL, pred_profit_pct REAL,
        pred_verdict TEXT, clf_verdict TEXT,
        confidence TEXT, ensemble_agreement INTEGER,
        star_power_used REAL, dir_power_used REAL)""")
    con.commit(); con.close()
    print("  ✓ Database ready →", DB_PATH)

def log_run(metrics):
    con = sqlite3.connect(DB_PATH)
    con.execute("""INSERT INTO training_runs
        (run_at,n_movies,reg_r2,reg_mae_cr,reg_cv_r2,clf_accuracy,
         best_reg_algo,best_clf_algo,ensemble_type,algo_scores)
        VALUES(?,?,?,?,?,?,?,?,?,?)""", (
        datetime.datetime.now().isoformat(),
        metrics["n_movies"], metrics["reg_r2"], metrics["reg_mae_cr"],
        metrics["reg_cv_r2"], metrics["clf_accuracy"],
        metrics["best_reg_algo"], metrics["best_clf_algo"],
        metrics["ensemble_type"], json.dumps(metrics["algo_scores"])
    ))
    con.commit(); con.close()

def save_pred(p):
    if not os.path.exists(DB_PATH): return
    con = sqlite3.connect(DB_PATH)
    con.execute("""INSERT INTO predictions
        (predicted_at,movie_name,star,director,language,genre,budget,opening_day,
         screens,release_month,release_year,is_franchise,
         pred_worldwide,pred_lower,pred_upper,pred_opening_wk,
         pred_profit,pred_profit_pct,pred_verdict,clf_verdict,
         confidence,ensemble_agreement,star_power_used,dir_power_used)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
        datetime.datetime.now().isoformat(),
        p["movie_name"],p["star"],p["director"],p["language"],p.get("genre","Unknown"),
        p["budget"],p["opening_day"],p["screens"],p["release_month"],p["release_year"],
        int(p["is_franchise"]),
        p["pred_worldwide"],p["pred_lower"],p["pred_upper"],p["pred_opening_wk"],
        p["pred_profit"],p["pred_profit_pct"],
        p["pred_verdict"],p["clf_verdict"],p["confidence"],
        int(p["ensemble_agreement"]),p["star_power_used"],p["dir_power_used"]
    ))
    con.commit(); con.close()

# ================================================================
#  HELPERS
# ================================================================
COLUMN_MAP = {
    "movie name":"Movie_Name","movie":"Movie_Name","movies":"Movie_Name","movie_name":"Movie_Name",
    "stars_featuring":"Star_Featuring","star_featuring":"Star_Featuring","star_power":"Star_Featuring",
    "director":"Director","language":"Language",
    "released_date":"Released_Date","released date":"Released_Date",
    "budget":"Budget","worldwide collection":"Worldwide","worldwide":"Worldwide",
    "india gross collection":"India_Gross","india gross":"India_Gross","india_gross":"India_Gross",
    "overseas collection":"Overseas","overseas":"Overseas","india_hindi_net":"India_Hindi_Net",
    "opening_day":"Opening_Day","screens":"Screens","verdict":"Verdict",
    "profit":"Profit_Raw","profit in percentage":"Profit_Pct_Raw",
    "rating":"Rating","genre":"Genre",
}
VERDICT_MAP = {
    "all time blockbuster":"BLOCKBUSTER",
    "blockbuster":         "BLOCKBUSTER",
    "super hit":           "SUPER HIT",
    "above average":       "HIT",
    "hit":                 "HIT",
    "average":             "AVERAGE",
    "below average":       "FLOP",
    "flop":                "FLOP",
    "disaster":            "FLOP",
}
GENRE_LIST = ["Action","Adventure","Biography","Comedy","Crime","Drama",
              "Family","Fantasy","Historical","Horror","Musical","Mystery",
              "Romance","Sci-Fi","Sports","Supernatural","Thriller"]

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

def load_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    rename = {c: COLUMN_MAP[c.lower().strip()] for c in df.columns if c.lower().strip() in COLUMN_MAP}
    return df.rename(columns=rename)

# ================================================================
#  SMOTE BALANCING UTILITY
# ================================================================
def apply_smote(X_train, y_train, strategy="auto", random_state=42):
    """
    Apply SMOTE to balance classifier training data.
    Falls back to SVMSMOTE if a class has too few samples,
    and uses RandomOverSampler as last resort.

    Parameters
    ----------
    X_train     : array-like of shape (n_samples, n_features)
    y_train     : array-like of shape (n_samples,)
    strategy    : sampling_strategy passed to SMOTE
    random_state: reproducibility seed

    Returns
    -------
    X_resampled, y_resampled, smote_report (dict)
    """
    from collections import Counter
    before = Counter(y_train.tolist() if hasattr(y_train, "tolist") else list(y_train))

    # SMOTE needs at least k_neighbors + 1 samples per minority class
    # Use k_neighbors = min(5, min_class_count - 1)  to be safe
    min_count = min(before.values())
    k_neighbors = max(1, min(5, min_count - 1))

    try:
        smote = SMOTE(
            sampling_strategy=strategy,
            k_neighbors=k_neighbors,
            random_state=random_state,
            n_jobs=-1
        )
        X_res, y_res = smote.fit_resample(X_train, y_train)
        method_used = f"SMOTE (k={k_neighbors})"
    except Exception as e1:
        print(f"    SMOTE failed ({e1}), trying SVMSMOTE...")
        try:
            smote = SVMSMOTE(
                sampling_strategy=strategy,
                k_neighbors=k_neighbors,
                random_state=random_state,
                n_jobs=-1
            )
            X_res, y_res = smote.fit_resample(X_train, y_train)
            method_used = f"SVMSMOTE (k={k_neighbors})"
        except Exception as e2:
            print(f"    SVMSMOTE failed ({e2}), using RandomOverSampler...")
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(sampling_strategy=strategy, random_state=random_state)
            X_res, y_res = ros.fit_resample(X_train, y_train)
            method_used = "RandomOverSampler"

    after = Counter(y_res.tolist() if hasattr(y_res, "tolist") else list(y_res))
    report = {
        "method": method_used,
        "before": dict(before),
        "after":  dict(after),
        "added":  len(y_res) - len(y_train),
    }
    return X_res, y_res, report


def print_smote_report(report, label_encoder):
    """Pretty-print the SMOTE balancing report with class names."""
    print(f"\n  ── SMOTE BALANCING REPORT ──")
    print(f"  Method used : {report['method']}")
    print(f"  Samples added: {report['added']}")
    print(f"\n  {'Class':<18} {'Before':>8}  {'After':>8}  {'Δ':>8}")
    print("  " + "-"*48)
    for cls_id in sorted(report['before'].keys()):
        cls_name = label_encoder.inverse_transform([cls_id])[0]
        b = report['before'].get(cls_id, 0)
        a = report['after'].get(cls_id, 0)
        print(f"  {cls_name:<18} {b:>8}  {a:>8}  {a-b:>+8}")
    print(f"  {'TOTAL':<18} {sum(report['before'].values()):>8}  "
          f"{sum(report['after'].values()):>8}  "
          f"{report['added']:>+8}")


# ================================================================
#  MAIN
# ================================================================
print("\n" + "="*65)
print("  INDIAN BOX OFFICE PREDICTOR  v5  (+SMOTE)")
print("  ✦ Rating + Log features  ✦ Confidence Intervals")
print("  ✦ Two-Stage Prediction   ✦ 12-Algo Benchmark")
print("  ✦ SMOTE Class Balancing  ✦ SQLite Logging")
print("="*65)

print("\n[0/10] Initialising database...")
init_db()

# ── Load ──────────────────────────────────────────────────────
print("\n[1/10] Loading datasets...")
files = sorted(glob.glob("data/*.csv"))
if not files:
    raise FileNotFoundError("No CSV files in data/.")

frames = []
for f in files:
    try:
        tmp = load_csv(f)
        frames.append(tmp)
        print(f"  ✓ {f}  ({tmp.shape[0]} rows)")
    except Exception as e:
        print(f"  ✗ {f}: {e}")

df = pd.concat(frames, ignore_index=True)
print(f"\n  Total rows: {df.shape[0]}")

for col in ["Movie_Name","Language","Director","Star_Featuring",
            "Budget","Worldwide","India_Gross","Overseas",
            "Opening_Day","Screens","Released_Date","Verdict","Genre","Rating"]:
    if col not in df.columns:
        df[col] = np.nan

# ── Clean ─────────────────────────────────────────────────────
print("\n[2/10] Cleaning...")
df["Star_Featuring"] = (df["Star_Featuring"].astype(str)
    .str.split(";").str[0].str.strip().replace("nan","Unknown"))
for col in ["Star_Featuring","Director","Language","Movie_Name"]:
    df[col] = df[col].fillna("Unknown")
df["Genre"] = df["Genre"].fillna("Drama")

for col in ["Budget","Worldwide","India_Gross","Overseas","Opening_Day","Screens","Rating"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Worldwide","Budget"])
df = df[(df["Budget"]>0) & (df["Worldwide"]>0)]
df["Opening_Day"] = df["Opening_Day"].fillna(df["Opening_Day"].median())
df["Screens"]     = df["Screens"].fillna(df["Screens"].median())
df["India_Gross"] = df["India_Gross"].fillna(df["India_Gross"].median())
df["Overseas"]    = df["Overseas"].fillna(df["Overseas"].median())
rating_median     = df["Rating"].median()
df["Rating"]      = df["Rating"].fillna(rating_median)
df.replace([np.inf,-np.inf], 0, inplace=True)
df = df.reset_index(drop=True)
q_low  = df["Worldwide"].quantile(0.01)
q_high = df["Worldwide"].quantile(0.99)
df = df[(df["Worldwide"] >= q_low) & (df["Worldwide"] <= q_high)]
df = df.reset_index(drop=True)
print(f"  Clean rows : {df.shape[0]}")
print(f"  Rating median used for missing: {rating_median:.1f}")

# ── Feature Engineering ───────────────────────────────────────
print("\n[3/10] Engineering features...")

df["Verdict_Clean"]     = (df["Verdict"].astype(str).str.strip().str.lower()
                            .map(VERDICT_MAP).fillna("AVERAGE"))
df["Log_Worldwide"]     = np.log1p(df["Worldwide"])
df["Profit"]            = df["Worldwide"] - df["Budget"]
df["Profit_Percentage"] = (df["Profit"] / df["Budget"]) * 100

df["Log_Budget"]        = np.log1p(df["Budget"])
df["Budget_Squared"]    = df["Budget"] ** 2
df["Overseas_Ratio"]    = df["Overseas"] / df["Worldwide"].replace(0, np.nan)
df["Overseas_Ratio"]    = df["Overseas_Ratio"].fillna(0).clip(0, 1)
df["Rating_x_Budget"]  = df["Rating"] * df["Budget"]

df["Opening_to_Budget"] = df["Opening_Day"] / df["Budget"]
df["Screens_to_Budget"] = df["Screens"] / df["Budget"]
df["Opening_per_Screen"]= df["Opening_Day"] / df["Screens"].replace(0, np.nan)
df.replace([np.inf,-np.inf], 0, inplace=True)
df["Opening_per_Screen"] = df["Opening_per_Screen"].fillna(0)

df["Released_Date"] = pd.to_datetime(df["Released_Date"], dayfirst=True, errors="coerce")
df["Release_Month"] = df["Released_Date"].dt.month.fillna(6).astype(int)
df["Release_Year"]  = df["Released_Date"].dt.year.fillna(2022).astype(int)
df["Season"]        = df["Release_Month"].apply(get_season)
df["Franchise"]     = df["Movie_Name"].astype(str).str.contains(
    r"\b(2|3|4|II|III|IV|Part|Chapter|Return|Reloaded|Revolution|Legacy)\b",
    case=False, regex=True).astype(int)

def festival_flag(row):
    m, d = row["Release_Month"], (row["Released_Date"].day
                                   if pd.notna(row["Released_Date"]) else 15)
    if m == 4 and d <= 25: return 1
    if m == 5 and d <= 15: return 1
    if m == 10 and d >= 20: return 1
    if m == 11 and d <= 10: return 1
    if m == 12 and d >= 20: return 1
    if m == 8 and 13 <= d <= 17: return 1
    return 0

df["Festival_Release"] = df.apply(festival_flag, axis=1)

label_language = LabelEncoder()
label_season   = LabelEncoder()
label_genre    = LabelEncoder()
df["Language_Label"] = label_language.fit_transform(df["Language"].astype(str))
df["Season_Label"]   = label_season.fit_transform(df["Season"])
df["Genre_Label"]    = label_genre.fit_transform(df["Genre"].astype(str))

print(f"  Languages     : {list(label_language.classes_)}")
print(f"  Genres        : {list(label_genre.classes_)}")
print(f"  Festival rows : {df['Festival_Release'].sum()}")
print(f"\n  Verdict distribution (raw):")
print(df["Verdict_Clean"].value_counts().to_string())

# ── Merge dynamic features if available ───────────────────────
DYNAMIC_PATH = "data/dynamic_features.csv"
if os.path.exists(DYNAMIC_PATH):
    print(f"\n  ✓ Found {DYNAMIC_PATH} — merging YouTube + Trends features...")
    dyn = pd.read_csv(DYNAMIC_PATH)
    dyn["Movie_Name"] = dyn["Movie_Name"].astype(str).str.strip()
    df = df.merge(dyn[["Movie_Name","Trailer_Views_M","Trends_Score","Trailer_Views_Log"]],
                  on="Movie_Name", how="left")
    df["Trailer_Views_M"]   = df["Trailer_Views_M"].fillna(df["Trailer_Views_M"].median())
    df["Trends_Score"]      = df["Trends_Score"].fillna(50)
    df["Trailer_Views_Log"] = df["Trailer_Views_Log"].fillna(df["Trailer_Views_Log"].median())

# ── Train/Test Split ──────────────────────────────────────────
print("\n[4/10] Splitting data...")

FEATURES = [
    "Budget", "Opening_Day", "Screens",
    "Language_Label", "Season_Label", "Franchise",
    "Opening_to_Budget", "Screens_to_Budget", "Opening_per_Screen",
    "Release_Year", "Genre_Label",
    "Rating", "Log_Budget", "Budget_Squared",
    "Overseas_Ratio", "Rating_x_Budget", "Festival_Release",
]

X     = df[FEATURES]
Y_reg = df["Log_Worldwide"]

X_train, X_test, _, _ = train_test_split(X, Y_reg, test_size=0.2, random_state=42)
train_idx, test_idx   = X_train.index, X_test.index
train_df = df.loc[train_idx].copy()
test_df  = df.loc[test_idx].copy()

star_power_map     = train_df.groupby("Star_Featuring")["Worldwide"].median()
director_power_map = train_df.groupby("Director")["Worldwide"].mean()
global_star_mean     = float(star_power_map.mean())
global_director_mean = float(director_power_map.mean())

for d in [train_df, test_df]:
    d["Star_Power"]     = d["Star_Featuring"].map(star_power_map).fillna(global_star_mean)
    d["Director_Power"] = d["Director"].map(director_power_map).fillna(global_director_mean)

FEATURES_FULL = FEATURES + ["Star_Power","Director_Power"]
X_train_full  = train_df[FEATURES_FULL]
X_test_full   = test_df[FEATURES_FULL]
Y_train_reg   = train_df["Log_Worldwide"]
Y_test_reg    = test_df["Log_Worldwide"]

label_encoder       = LabelEncoder()
df["Verdict_Label"] = label_encoder.fit_transform(df["Verdict_Clean"])
Y_clf_train = df.loc[X_train_full.index, "Verdict_Label"]
Y_clf_test  = df.loc[X_test_full.index,  "Verdict_Label"]

# ── SMOTE Balancing ───────────────────────────────────────────
# NOTE: SMOTE is applied ONLY to classifier training data
#       Test data is NEVER resampled (avoids data leakage)
print("\n[4b/10] Applying SMOTE to balance classifier training data...")

X_train_smote, Y_clf_train_smote, smote_report = apply_smote(
    X_train_full.values, Y_clf_train.values,
    strategy="auto",   # upsample all minority classes to match majority
    random_state=42
)
print_smote_report(smote_report, label_encoder)

# Convert back to DataFrame for consistency
X_train_smote_df = pd.DataFrame(X_train_smote, columns=FEATURES_FULL)

# Recalculate sample weights on SMOTE-balanced data (still useful for some algos)
counts_sm  = Counter(Y_clf_train_smote.tolist())
n_total_sm = len(Y_clf_train_smote)
n_cls_sm   = len(counts_sm)
sw_sm      = {cls: n_total_sm/(n_cls_sm*cnt) for cls,cnt in counts_sm.items()}
sample_weights_smote = np.array([sw_sm[y] for y in Y_clf_train_smote])

scaler   = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_train_full)      # scaled original (for regressors)
X_te_sc  = scaler.transform(X_test_full)

scaler_sm   = StandardScaler()
X_tr_sm_sc  = scaler_sm.fit_transform(X_train_smote_df)  # scaled SMOTE (for linear classifiers)

print(f"\n  Train: {len(X_train_full)} (original) → "
      f"{len(X_train_smote_df)} (after SMOTE) | Test: {len(X_test_full)}")
print(f"  Features: {len(FEATURES_FULL)}")

# ================================================================
#  STEP 5 — TWO-STAGE PREDICTION
# ================================================================
print("\n[5/10] Training two-stage predictor...")

STAGE1_FEATURES = [
    "Budget", "Screens", "Language_Label", "Season_Label",
    "Franchise", "Screens_to_Budget", "Release_Year",
    "Genre_Label", "Log_Budget", "Budget_Squared",
    "Rating", "Festival_Release", "Star_Power", "Director_Power",
]

df["Opening_Week"]     = df["Opening_Day"] * 3.2
df["Log_Opening_Week"] = np.log1p(df["Opening_Week"])
train_df["Opening_Week"]     = train_df["Opening_Day"] * 3.2
train_df["Log_Opening_Week"] = np.log1p(train_df["Opening_Week"])
test_df["Opening_Week"]      = test_df["Opening_Day"] * 3.2
test_df["Log_Opening_Week"]  = np.log1p(test_df["Opening_Week"])

Y_s1_train = train_df["Log_Opening_Week"]
Y_s1_test  = test_df["Log_Opening_Week"]

stage1_model = XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
)
stage1_model.fit(train_df[STAGE1_FEATURES], Y_s1_train)

train_df["Pred_Opening_Week"] = np.expm1(stage1_model.predict(train_df[STAGE1_FEATURES]))
test_df["Pred_Opening_Week"]  = np.expm1(stage1_model.predict(test_df[STAGE1_FEATURES]))

FEATURES_S2 = FEATURES_FULL + ["Pred_Opening_Week", "Log_Opening_Week"]

X_train_s2 = train_df[FEATURES_S2]
X_test_s2  = test_df[FEATURES_S2]

s1_r2 = r2_score(Y_s1_test, stage1_model.predict(test_df[STAGE1_FEATURES]))
print(f"  Stage-1 R² (opening week) : {s1_r2:.4f}")

# ── Extend SMOTE-balanced data with Stage-2 features ─────────
# We add Pred_Opening_Week and Log_Opening_Week to SMOTE train set.
# These are computed from Stage-1 predictions on original train data,
# then the SMOTE synthetic samples inherit interpolated values.
train_df_s2_vals = train_df[FEATURES_S2].values
X_train_s2_smote, Y_clf_s2_smote, smote_report_s2 = apply_smote(
    train_df_s2_vals, Y_clf_train.values,
    strategy="auto", random_state=42
)
X_train_s2_smote_df = pd.DataFrame(X_train_s2_smote, columns=FEATURES_S2)

print(f"\n  Stage-2 SMOTE: {len(X_train_s2)} → {len(X_train_s2_smote_df)} samples")

# Scalers for linear models (using s2 + SMOTE)
scaler_s2     = StandardScaler()
X_tr_s2_sc    = scaler_s2.fit_transform(X_train_s2)          # original (for regressors)
X_te_s2_sc    = scaler_s2.transform(X_test_s2)

scaler_s2_sm  = StandardScaler()
X_tr_s2_sm_sc = scaler_s2_sm.fit_transform(X_train_s2_smote_df)  # SMOTE (for linear clfs)

# Sample weights on s2-SMOTE data
counts_s2  = Counter(Y_clf_s2_smote.tolist())
sw_s2      = {cls: len(Y_clf_s2_smote)/(len(counts_s2)*cnt) for cls,cnt in counts_s2.items()}
sample_weights_s2 = np.array([sw_s2[y] for y in Y_clf_s2_smote])

# ================================================================
#  STEP 6 — BENCHMARK 12 ALGORITHMS
# ================================================================
print("\n[6/10] Benchmarking 12 algorithms...")
print("  (takes ~4-8 minutes)\n")

LINEAR_REGS = {"Ridge","Lasso","ElasticNet","BayesianRidge","KNN","SVR"}
LINEAR_CLFS = {"LogisticRegression","KNN","SVM","NaiveBayes"}

REGRESSORS = {
    "XGBoost":          XGBRegressor(n_estimators=600,learning_rate=0.05,max_depth=6,
                            subsample=0.8,colsample_bytree=0.8,random_state=42,verbosity=0),
    "LightGBM":         (LGBMRegressor(n_estimators=600,learning_rate=0.05,max_depth=6,
                            random_state=42,verbose=-1,n_jobs=-1)
                         if HAS_LGBM else ExtraTreesRegressor(n_estimators=400,random_state=42,n_jobs=-1)),
    "CatBoost":         (CatBoostRegressor(iterations=400,learning_rate=0.05,depth=6,
                            random_seed=42,verbose=0)
                         if HAS_CATBOOST else GradientBoostingRegressor(n_estimators=300,
                            learning_rate=0.05,max_depth=5,random_state=42)),
    "RandomForest":     RandomForestRegressor(n_estimators=500,max_depth=10,
                            min_samples_leaf=2,random_state=42,n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=300,learning_rate=0.05,
                            max_depth=5,subsample=0.8,random_state=42),
    "ExtraTrees":       ExtraTreesRegressor(n_estimators=500,max_depth=10,
                            random_state=42,n_jobs=-1),
    "AdaBoost":         AdaBoostRegressor(n_estimators=200,learning_rate=0.05,random_state=42),
    "Ridge":            Ridge(alpha=1.0),
    "Lasso":            Lasso(alpha=0.01,max_iter=5000),
    "ElasticNet":       ElasticNet(alpha=0.01,l1_ratio=0.5,max_iter=5000),
    "KNN":              KNeighborsRegressor(n_neighbors=7,n_jobs=-1),
    "SVR":              SVR(C=10,gamma="scale",kernel="rbf"),
}

reg_scores = {}
print(f"  {'Algorithm':<22} {'Test R²':>8}  {'MAE(Cr)':>9}  {'CV R²':>8}")
print("  " + "-"*55)

for name, model in REGRESSORS.items():
    try:
        # Regressors use ORIGINAL (unbalanced) training data — correct for regression
        Xtr = X_tr_s2_sc if name in LINEAR_REGS else X_train_s2.values
        Xte = X_te_s2_sc if name in LINEAR_REGS else X_test_s2.values
        model.fit(Xtr, Y_train_reg)
        preds  = model.predict(Xte)
        r2v    = r2_score(Y_test_reg, preds)
        mae_cr = mean_absolute_error(np.expm1(Y_test_reg), np.expm1(preds))
        n_cv   = 3 if name in {"SVR","KNN","AdaBoost"} else 5
        cv_r2  = cross_val_score(model,
                    X_tr_s2_sc if name in LINEAR_REGS else X_train_s2.values,
                    Y_train_reg,
                    cv=KFold(n_splits=n_cv,shuffle=True,random_state=42),
                    scoring="r2",n_jobs=-1).mean()
        reg_scores[name] = {"r2":r2v,"mae_cr":mae_cr,"cv_r2":cv_r2,"model":model}
        print(f"  {name:<22} {r2v:>8.4f}  {mae_cr:>9.1f}  {cv_r2:>8.4f}")
    except Exception as e:
        print(f"  {name:<22} ERROR: {e}")
        reg_scores[name] = {"r2":0,"mae_cr":9999,"cv_r2":0,"model":model}

# ── Classifiers — use SMOTE-balanced data ────────────────────
CLASSIFIERS = {
    "XGBoost":            XGBClassifier(n_estimators=600,learning_rate=0.05,max_depth=6,
                              subsample=0.8,colsample_bytree=0.8,
                              eval_metric="mlogloss",random_state=42,verbosity=0),
    "LightGBM":           (LGBMClassifier(n_estimators=600,learning_rate=0.05,max_depth=6,
                              class_weight="balanced",random_state=42,verbose=-1,n_jobs=-1)
                           if HAS_LGBM else ExtraTreesClassifier(n_estimators=400,
                              class_weight="balanced",random_state=42,n_jobs=-1)),
    "CatBoost":           (CatBoostClassifier(iterations=400,learning_rate=0.05,depth=6,
                              auto_class_weights="Balanced",random_seed=42,verbose=0)
                           if HAS_CATBOOST else GradientBoostingClassifier(n_estimators=300,
                              learning_rate=0.05,max_depth=5,random_state=42)),
    "RandomForest":       RandomForestClassifier(n_estimators=500,max_depth=10,
                              class_weight="balanced",random_state=42,n_jobs=-1),
    "GradientBoosting":   GradientBoostingClassifier(n_estimators=300,learning_rate=0.05,
                              max_depth=5,subsample=0.8,random_state=42),
    "ExtraTrees":         ExtraTreesClassifier(n_estimators=500,max_depth=10,
                              class_weight="balanced",random_state=42,n_jobs=-1),
    "AdaBoost":           AdaBoostClassifier(n_estimators=200,learning_rate=0.05,random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=2000,class_weight="balanced",C=1.0,random_state=42),
    "DecisionTree":       DecisionTreeClassifier(max_depth=8,class_weight="balanced",random_state=42),
    "KNN":                KNeighborsClassifier(n_neighbors=7,n_jobs=-1),
    "SVM":                SVC(C=10,kernel="rbf",class_weight="balanced",probability=True,random_state=42),
    "NaiveBayes":         GaussianNB(),
}

clf_scores = {}
NEEDS_SW   = {"XGBoost","GradientBoosting","AdaBoost","RandomForest",
              "ExtraTrees","LightGBM","CatBoost","DecisionTree"}

print(f"\n  Classifiers use SMOTE-balanced training data ({len(X_train_s2_smote_df)} samples)")
print(f"  {'Algorithm':<22} {'Test Acc':>10}  {'CV Acc':>8}")
print("  " + "-"*44)

for name, model in CLASSIFIERS.items():
    try:
        # SMOTE-balanced data for classifiers
        Xtr = X_tr_s2_sm_sc if name in LINEAR_CLFS else X_train_s2_smote_df.values
        # Test data stays original (no SMOTE on test set)
        Xte = X_te_s2_sc    if name in LINEAR_CLFS else X_test_s2.values

        fit_kw = {"sample_weight": sample_weights_s2} if name in NEEDS_SW else {}
        model.fit(Xtr, Y_clf_s2_smote, **fit_kw)
        preds  = model.predict(Xte)
        acc    = accuracy_score(Y_clf_test, preds)
        n_cv   = 3 if name in {"SVM","KNN","AdaBoost"} else 5
        cv_acc = cross_val_score(model,
                    Xtr, Y_clf_s2_smote,
                    cv=StratifiedKFold(n_splits=n_cv,shuffle=True,random_state=42),
                    scoring="accuracy",n_jobs=-1).mean()
        clf_scores[name] = {"acc":acc,"cv_acc":cv_acc,"model":model}
        print(f"  {name:<22} {acc:>10.4f}  {cv_acc:>8.4f}")
    except Exception as e:
        print(f"  {name:<22} ERROR: {e}")
        clf_scores[name] = {"acc":0,"cv_acc":0,"model":model}

# ================================================================
#  STEP 7 — AUTO-SELECT TOP 3 → STACK
# ================================================================
print("\n[7/10] Auto-selecting top 3 → stacking ensemble...")

top3_reg = sorted(reg_scores.items(),
    key=lambda x: 0.6 * x[1]["cv_r2"] + 0.4 * x[1]["r2"],
    reverse=True)[:3]
top3_clf = sorted(clf_scores.items(), key=lambda x: x[1]["cv_acc"], reverse=True)[:3]

print(f"\n  ★ Top 3 Regressors  : {[n for n,_ in top3_reg]}")
print(f"  ★ Top 3 Classifiers : {[n for n,_ in top3_clf]}")

reg_base = [(n.lower().replace(" ","_"), reg_scores[n]["model"]) for n,_ in top3_reg]
clf_base = [(n.lower().replace(" ","_"), clf_scores[n]["model"]) for n,_ in top3_clf]

# ─────────────────────────────────────────────
#  IMPROVED STACKING (ALOHA UPGRADE)
# ─────────────────────────────────────────────

# Better meta learner for regression
final_regressor = StackingRegressor(
    estimators=reg_base,
    final_estimator=ElasticNet(alpha=0.01, l1_ratio=0.7),
    cv=5,
    n_jobs=-1,
    passthrough=True
)

# Better meta learner for classification
meta_clf = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    C=2.0
)

final_classifier = StackingClassifier(
    estimators=clf_base,
    final_estimator=meta_clf,
    cv=5,
    n_jobs=-1,
    passthrough=True
)

# Calibration (BIG BOOST)
final_classifier = CalibratedClassifierCV(
    final_classifier,
    method='sigmoid',
    cv=3
)

print("  Training stacking regressor (original data)...")
final_regressor.fit(X_train_s2, Y_train_reg)

print("  Training stacking classifier (SMOTE-balanced data)...")
final_classifier.fit(X_train_s2_smote_df, Y_clf_s2_smote)

Y_pred_reg = final_regressor.predict(X_test_s2)
Y_pred_clf = final_classifier.predict(X_test_s2)

# ─────────────────────────────────────────────
#  ALOHA Weighted Ensemble (FINAL BOOST)
# ─────────────────────────────────────────────

print("\n[Extra] ALOHA Weighted Ensemble...")

top_models = [clf_scores[n]["model"] for n, _ in top3_clf]

preds = []
weights = []

for name, _ in top3_clf:
    model = clf_scores[name]["model"]
    pred = model.predict(X_test_s2)
    preds.append(pred)
    weights.append(clf_scores[name]["cv_acc"])

preds = np.array(preds)
weights = np.array(weights)

final_weighted = []

for i in range(preds.shape[1]):
    votes = {}
    for j in range(len(preds)):
        cls = preds[j][i]
        votes[cls] = votes.get(cls, 0) + weights[j]
    final_weighted.append(max(votes, key=votes.get))

final_weighted = np.array(final_weighted)

aloha_acc = accuracy_score(Y_clf_test, final_weighted)

print(f"  ALOHA Ensemble Accuracy : {aloha_acc:.4f}")

r2     = r2_score(Y_test_reg, Y_pred_reg)
mae    = mean_absolute_error(Y_test_reg, Y_pred_reg)
mae_cr = mean_absolute_error(np.expm1(Y_test_reg), np.expm1(Y_pred_reg))
acc    = accuracy_score(Y_clf_test, Y_pred_clf)

df["Star_Power_Full"]     = df["Star_Featuring"].map(star_power_map).fillna(global_star_mean)
df["Director_Power_Full"] = df["Director"].map(director_power_map).fillna(global_director_mean)
df["Star_Power"]     = df["Star_Featuring"].map(star_power_map).fillna(global_star_mean)
df["Director_Power"] = df["Director"].map(director_power_map).fillna(global_director_mean)
df["Pred_Opening_Week"]   = np.expm1(stage1_model.predict(df[STAGE1_FEATURES]))
df["Log_Opening_Week"]    = np.log1p(df["Opening_Day"] * 3.2)
FEATURES_CV = FEATURES + ["Star_Power_Full","Director_Power_Full",
                           "Pred_Opening_Week","Log_Opening_Week"]

kf    = KFold(n_splits=5,shuffle=True,random_state=42)
cv_r2 = cross_val_score(reg_scores[top3_reg[0][0]]["model"],
    df[FEATURES_CV].values, df["Log_Worldwide"],
    cv=kf, scoring="r2", n_jobs=-1).mean()

print(f"\n  ── FINAL ENSEMBLE RESULTS ──")
print(f"  R² Score       : {r2:.4f}")
print(f"  MAE (log)      : {mae:.4f}")
print(f"  MAE (Crores)   : ₹{mae_cr:.1f} Cr")
print(f"  CV R² (best)   : {cv_r2:.4f}")
print(f"  Clf Accuracy   : {acc:.4f}")

print(f"\n  ── REGRESSOR LEADERBOARD ──")
print(f"  {'Rank':<5}{'Algorithm':<22}{'R²':>8}  {'MAE Cr':>8}  {'CV R²':>8}")
print("  " + "-"*56)
for i,(n,s) in enumerate(sorted(reg_scores.items(),key=lambda x:x[1]["cv_r2"],reverse=True),1):
    print(f"  {i:<5}{n:<22}{s['r2']:>8.4f}  {s['mae_cr']:>8.1f}  {s['cv_r2']:>8.4f}"
          + (" ★" if i<=3 else ""))

# ── Hyperparameter Tuning ─────────────────────────────────────
print("\n  Tuning CatBoost hyperparameters...")
catboost_params = {
    "iterations":    randint(200, 600),
    "learning_rate": uniform(0.01, 0.09),
    "depth":         randint(4, 8),
}
cat_search = RandomizedSearchCV(
    CatBoostRegressor(random_seed=42, verbose=0),
    param_distributions=catboost_params,
    n_iter=40, cv=3, scoring="r2", random_state=42, n_jobs=-1,
)
cat_search.fit(X_train_s2, Y_train_reg)
print(f"  ✓ Best CatBoost params: {cat_search.best_params_}")
REGRESSORS["CatBoost"] = cat_search.best_estimator_
reg_scores["CatBoost"]["model"] = cat_search.best_estimator_

print("\n  Tuning XGBoost hyperparameters...")
xgb_params = {
    "n_estimators":     randint(400, 800),
    "learning_rate":    uniform(0.01, 0.09),
    "max_depth":        randint(4, 8),
    "subsample":        uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
}
xgb_search = RandomizedSearchCV(
    XGBRegressor(random_state=42, verbosity=0),
    param_distributions=xgb_params,
    n_iter=20, cv=3, scoring="r2", random_state=42, n_jobs=-1,
)
xgb_search.fit(X_train_s2, Y_train_reg)
print(f"  ✓ Best XGBoost params : {xgb_search.best_params_}")
REGRESSORS["XGBoost"] = xgb_search.best_estimator_
reg_scores["XGBoost"]["model"] = xgb_search.best_estimator_

print(f"\n  ── CLASSIFIER LEADERBOARD ──")
print(f"  {'Rank':<5}{'Algorithm':<22}{'Acc':>10}  {'CV Acc':>8}")
print("  " + "-"*48)
for i,(n,s) in enumerate(sorted(clf_scores.items(),key=lambda x:x[1]["cv_acc"],reverse=True),1):
    print(f"  {i:<5}{n:<22}{s['acc']:>10.4f}  {s['cv_acc']:>8.4f}"
          + (" ★" if i<=3 else ""))

# ================================================================
#  STEP 8 — CONFIDENCE INTERVALS (Quantile Regression)
# ================================================================
print("\n[8/10] Training confidence interval models...")

lower_model = GradientBoostingRegressor(
    loss="quantile", alpha=0.10,
    n_estimators=300, learning_rate=0.05, max_depth=5,
    subsample=0.8, random_state=42
)
upper_model = GradientBoostingRegressor(
    loss="quantile", alpha=0.90,
    n_estimators=300, learning_rate=0.05, max_depth=5,
    subsample=0.8, random_state=42
)
# Quantile regressors use original (unbalanced) training data
lower_model.fit(X_train_s2, Y_train_reg)
upper_model.fit(X_train_s2, Y_train_reg)

lower_preds = np.expm1(lower_model.predict(X_test_s2))
upper_preds = np.expm1(upper_model.predict(X_test_s2))
actual_cr   = np.expm1(Y_test_reg.values)
coverage    = np.mean((actual_cr >= lower_preds) & (actual_cr <= upper_preds))
avg_width   = np.mean(upper_preds - lower_preds)

print(f"  ✓ Interval coverage : {coverage*100:.1f}%  (target ~70%)")
print(f"  ✓ Average width     : ₹{avg_width:.0f} Cr")

# ================================================================
#  STEP 9 — SAVE ALL MODELS
# ================================================================
print("\n[9/10] Saving models...")
os.makedirs("models", exist_ok=True)

joblib.dump(final_regressor,      "models/regressor.pkl")
joblib.dump(final_classifier,     "models/classifier.pkl")
joblib.dump(stage1_model,         "models/stage1_model.pkl")
joblib.dump(lower_model,          "models/lower_bound.pkl")
joblib.dump(upper_model,          "models/upper_bound.pkl")
joblib.dump(label_encoder,        "models/label_encoder.pkl")
joblib.dump(label_language,       "models/label_language.pkl")
joblib.dump(label_season,         "models/label_season.pkl")
joblib.dump(label_genre,          "models/label_genre.pkl")
joblib.dump(star_power_map,       "models/star_power_map.pkl")
joblib.dump(director_power_map,   "models/director_power_map.pkl")
joblib.dump(scaler,               "models/scaler.pkl")
joblib.dump(scaler_s2,            "models/scaler_s2.pkl")
joblib.dump(smote_report_s2,      "models/smote_report.pkl")   # ← NEW: saved for inspection
joblib.dump({
    "global_star_mean":     global_star_mean,
    "global_director_mean": global_director_mean,
    "features_full":        FEATURES_FULL,
    "features_s2":          FEATURES_S2,
    "stage1_features":      STAGE1_FEATURES,
    "top3_reg":             [n for n,_ in top3_reg],
    "top3_clf":             [n for n,_ in top3_clf],
    "reg_r2": r2, "reg_mae_cr": mae_cr, "clf_accuracy": acc,
    "interval_coverage": coverage, "interval_avg_width": avg_width,
    "rating_median": rating_median,
    "smote_method":   smote_report_s2["method"],
    "smote_added":    smote_report_s2["added"],
    "smote_before":   smote_report_s2["before"],
    "smote_after":    smote_report_s2["after"],
    "reg_scores": {k:{"r2":v["r2"],"mae_cr":v["mae_cr"],"cv_r2":v["cv_r2"]}
                   for k,v in reg_scores.items()},
    "clf_scores": {k:{"acc":v["acc"],"cv_acc":v["cv_acc"]}
                   for k,v in clf_scores.items()},
}, "models/meta.pkl")

print("  ✓ All models saved (incl. smote_report.pkl)")
log_run({
    "n_movies": df.shape[0], "reg_r2": r2, "reg_mae_cr": mae_cr, "reg_cv_r2": cv_r2,
    "clf_accuracy": acc, "best_reg_algo": top3_reg[0][0], "best_clf_algo": top3_clf[0][0],
    "ensemble_type": f"TwoStage+SMOTE+Stack({','.join(n for n,_ in top3_reg)})→Ridge",
    "algo_scores": {
        "reg": {k:{"r2":v["r2"],"mae_cr":v["mae_cr"],"cv_r2":v["cv_r2"]}
                for k,v in reg_scores.items()},
        "clf": {k:{"acc":v["acc"],"cv_acc":v["cv_acc"]}
                for k,v in clf_scores.items()},
    }
})
print("  ✓ Training run logged to database")

# ================================================================
#  STEP 10 — CHARTS
# ================================================================
print("\n[10/10] Generating charts...")

predicted_cr = np.expm1(Y_pred_reg)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Indian Box Office Predictor v5 — Two-Stage + SMOTE + Confidence Intervals",
             fontsize=14, fontweight="bold")

# Chart 1 — Actual vs Predicted with confidence interval
ax = axes[0,0]
sorted_idx = np.argsort(actual_cr)
ax.fill_between(range(len(sorted_idx)),
                lower_preds[sorted_idx], upper_preds[sorted_idx],
                alpha=0.25, color="#7F77DD", label="70% confidence interval")
ax.scatter(range(len(sorted_idx)), actual_cr[sorted_idx],
           s=20, color="#D85A30", alpha=0.7, label="Actual")
ax.scatter(range(len(sorted_idx)), predicted_cr[sorted_idx],
           s=20, color="#7F77DD", alpha=0.7, label="Predicted")
ax.set_xlabel("Movies (sorted by actual)"); ax.set_ylabel("Collection (Cr)")
ax.set_title(f"Actual vs Predicted\nR² = {r2:.3f}  |  Coverage = {coverage*100:.0f}%")
ax.legend(fontsize=8)

# Chart 2 — Regressor leaderboard
ax = axes[0,1]
rnames = [n for n,_ in sorted(reg_scores.items(),key=lambda x:x[1]["cv_r2"],reverse=True)]
rvals  = [reg_scores[n]["cv_r2"] for n in rnames]
rcols  = ["#7F77DD" if n in [x for x,_ in top3_reg] else "#B4B2A9" for n in rnames]
ax.barh(rnames, rvals, color=rcols)
ax.axvline(r2, color="red", linestyle="--", lw=1.2, label=f"Ensemble={r2:.3f}")
ax.set_xlabel("CV R²"); ax.set_title("Regressor Leaderboard\nPurple = in ensemble")
ax.legend(fontsize=8)

# Chart 3 — Classifier leaderboard
ax = axes[0,2]
cnames = [n for n,_ in sorted(clf_scores.items(),key=lambda x:x[1]["cv_acc"],reverse=True)]
cvals  = [clf_scores[n]["cv_acc"] for n in cnames]
ccols  = ["#7F77DD" if n in [x for x,_ in top3_clf] else "#B4B2A9" for n in cnames]
ax.barh(cnames, cvals, color=ccols)
ax.axvline(acc, color="red", linestyle="--", lw=1.2, label=f"Ensemble={acc:.3f}")
ax.set_xlabel("CV Accuracy"); ax.set_title("Classifier Leaderboard (SMOTE-trained)\nPurple = in ensemble")
ax.legend(fontsize=8)

# Chart 4 — Confusion matrix
ax = axes[1,0]
cm = confusion_matrix(Y_clf_test, Y_pred_clf)
sns.heatmap(cm, annot=True, fmt="d", ax=ax,
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_, cmap="Purples")
ax.set_title(f"Confusion Matrix\nAccuracy = {acc:.3f}")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=7)

# Chart 5 — SMOTE Before vs After class distribution
ax = axes[1,1]
classes    = [label_encoder.inverse_transform([c])[0]
              for c in sorted(smote_report_s2["before"].keys())]
before_cnt = [smote_report_s2["before"].get(c, 0)
              for c in sorted(smote_report_s2["before"].keys())]
after_cnt  = [smote_report_s2["after"].get(c, 0)
              for c in sorted(smote_report_s2["after"].keys())]
x = np.arange(len(classes)); w = 0.35
bars1 = ax.bar(x - w/2, before_cnt, w, label="Before SMOTE", color="#D85A30", alpha=0.8)
bars2 = ax.bar(x + w/2, after_cnt,  w, label="After SMOTE",  color="#7F77DD", alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(classes, rotation=25, ha="right", fontsize=8)
ax.set_ylabel("Sample Count")
ax.set_title(f"SMOTE Class Balancing\n({smote_report_s2['method']}  +{smote_report_s2['added']} samples)")
ax.legend(fontsize=8)
for bar in bars1: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                           str(int(bar.get_height())), ha='center', va='bottom', fontsize=7)
for bar in bars2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                           str(int(bar.get_height())), ha='center', va='bottom', fontsize=7)

# Chart 6 — Residuals
ax = axes[1,2]
residuals = actual_cr - predicted_cr
ax.scatter(predicted_cr, residuals, alpha=0.5, s=15, color="#7F77DD")
ax.axhline(0, color="red", linestyle="--", lw=1.2)
ax.set_xlabel("Predicted (Cr)"); ax.set_ylabel("Residual (Cr)")
ax.set_title("Residual Plot\n(Actual − Predicted)")

fig.tight_layout()
fig.savefig("models/results.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Chart saved → models/results.png  (6-panel, incl. SMOTE chart)")

# ── EXTRA: Standalone Classifier Leaderboard (separate PNG) ──
fig2, ax2 = plt.subplots(figsize=(7, 6))

cnames_sorted = [n for n, _ in sorted(clf_scores.items(),
                  key=lambda x: x[1]["acc"], reverse=False)]  # ascending for barh
cvals_sorted  = [clf_scores[n]["acc"] for n in cnames_sorted]
ccols_sorted  = ["#7F77DD" if n in [x for x, _ in top3_clf]
                 else "#B4B2A9" for n in cnames_sorted]

bars = ax2.barh(cnames_sorted, cvals_sorted,
                color=ccols_sorted, edgecolor="white", height=0.7)

# Ensemble line — uses ALOHA accuracy
ax2.axvline(x=aloha_acc, color="red", linestyle="--", linewidth=1.5,
            label=f"Ensemble={aloha_acc:.3f}")

# Value labels on bars
for bar, val in zip(bars, cvals_sorted):
    ax2.text(bar.get_width() + 0.003,
             bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", ha="left",
             fontsize=8, color="#333333")

ax2.set_xlabel("CV Accuracy", fontsize=10)
ax2.set_xlim(0, max(cvals_sorted) + 0.12)
ax2.set_title("Classifier Leaderboard\nPurple = in ensemble",
              fontsize=11, fontweight="bold")
ax2.legend(loc="lower right", fontsize=9)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.tick_params(axis="y", labelsize=9)
ax2.tick_params(axis="x", labelsize=9)

fig2.tight_layout()
fig2.savefig("models/classifier_leaderboard.png", dpi=150,
             bbox_inches="tight", facecolor="white")
plt.show()
print("  ✓ Classifier leaderboard saved → models/classifier_leaderboard.png")

# ================================================================
#  TERMINAL PREDICTION
# ================================================================
print("\n" + "="*65)
print("  PREDICT A MOVIE  (Ctrl+C to skip)")
print("="*65)

try:
    movie_name    = input("\nMovie Name               : ").strip()
    star_name     = input("Star Featuring           : ").strip().title()
    director_name = input("Director                 : ").strip().title()
    budget        = float(input("Budget (Cr)              : "))
    opening_day   = float(input("Opening Day Est. (Cr)    : "))
    screens       = int(input("Worldwide Screens        : "))
    language      = input("Language                 : ").strip().title()
    genre_input   = input(f"Genre                    : ").strip().title()
    rating        = float(input("Expected Rating (0-10)   : ") or "6.5")
    release_month = int(input("Release Month (1-12)     : "))
    release_year  = int(input("Release Year             : ") or "2025")
    is_franchise  = input("Sequel/Franchise? (y/n)  : ").strip().lower() == "y"

    star_match = get_close_matches(star_name, star_power_map.index.tolist(), n=1, cutoff=0.6)
    star_val   = float(star_power_map[star_match[0]]) if star_match else global_star_mean
    print(f"  → Star    : {'Matched ' + star_match[0] if star_match else 'using average'}")

    dir_match  = get_close_matches(director_name, director_power_map.index.tolist(), n=1, cutoff=0.6)
    dir_val    = float(director_power_map[dir_match[0]]) if dir_match else global_director_mean
    print(f"  → Director: {'Matched ' + dir_match[0] if dir_match else 'using average'}")

    lang_val   = int(label_language.transform([language])[0]) if language in label_language.classes_ else 0
    season_str = get_season(release_month)
    season_val = int(label_season.transform([season_str])[0]) if season_str in label_season.classes_ else 0
    genre_val  = int(label_genre.transform([genre_input])[0]) if genre_input in label_genre.classes_ else 0
    fest_val   = 1 if release_month in [4,5,10,11,12] else 0
    log_bud    = np.log1p(budget)
    bud_sq     = budget ** 2

    s1_row = pd.DataFrame([{
        "Budget":budget,"Screens":screens,"Language_Label":lang_val,"Season_Label":season_val,
        "Franchise":int(is_franchise),"Screens_to_Budget":screens/budget,"Release_Year":release_year,
        "Genre_Label":genre_val,"Log_Budget":log_bud,"Budget_Squared":bud_sq,
        "Rating":rating,"Festival_Release":fest_val,"Star_Power":star_val,"Director_Power":dir_val,
    }])
    pred_opening_wk = round(float(np.expm1(stage1_model.predict(s1_row)[0])), 1)

    s2_row = pd.DataFrame([{
        "Budget":budget,"Opening_Day":opening_day,"Screens":screens,
        "Language_Label":lang_val,"Season_Label":season_val,"Franchise":int(is_franchise),
        "Opening_to_Budget":opening_day/budget,"Screens_to_Budget":screens/budget,
        "Opening_per_Screen":opening_day/screens if screens>0 else 0,
        "Release_Year":release_year,"Genre_Label":genre_val,
        "Rating":rating,"Log_Budget":log_bud,"Budget_Squared":bud_sq,
        "Overseas_Ratio":0.15,"Rating_x_Budget":rating*budget,"Festival_Release":fest_val,
        "Star_Power":star_val,"Director_Power":dir_val,
        "Pred_Opening_Week":pred_opening_wk,"Log_Opening_Week":np.log1p(pred_opening_wk),
    }])

    pred_log       = final_regressor.predict(s2_row)[0]
    pred_worldwide = round(float(np.expm1(pred_log)), 1)
    pred_lower     = round(float(np.expm1(lower_model.predict(s2_row)[0])), 1)
    pred_upper     = round(float(np.expm1(upper_model.predict(s2_row)[0])), 1)
    clf_id         = final_classifier.predict(s2_row)[0]
    clf_verdict    = label_encoder.inverse_transform([clf_id])[0]

    profit        = pred_worldwide - budget
    profit_pct    = (profit / budget) * 100
    final_verdict = verdict_from_profit(profit_pct)
    agree         = (clf_verdict == final_verdict)
    confidence    = "HIGH (both agree)" if agree else f"MEDIUM (classifier: {clf_verdict})"

    print("\n" + "="*65)
    print("  FINAL PREDICTION (v5 — Two-Stage + SMOTE + Confidence Interval)")
    print("="*65)
    print(f"  Movie              : {movie_name}")
    print(f"  Predicted Opening  : ₹{pred_opening_wk} Cr  (week 1)")
    print(f"  Predicted Lifetime : ₹{pred_worldwide} Cr")
    print(f"  Confidence Range   : ₹{pred_lower} Cr  →  ₹{pred_upper} Cr")
    print(f"  Budget             : ₹{budget} Cr")
    print(f"  Profit             : ₹{round(profit,1)} Cr  ({round(profit_pct,1)}%)")
    print(f"  Verdict            : {final_verdict}")
    print(f"  Confidence         : {confidence}")
    print("="*65)

    save_pred({
        "movie_name":movie_name,"star":star_name,"director":director_name,
        "language":language,"genre":genre_input,"budget":budget,"opening_day":opening_day,
        "screens":screens,"release_month":release_month,"release_year":release_year,
        "is_franchise":is_franchise,"pred_worldwide":pred_worldwide,
        "pred_lower":pred_lower,"pred_upper":pred_upper,"pred_opening_wk":pred_opening_wk,
        "pred_profit":profit,"pred_profit_pct":profit_pct,
        "pred_verdict":final_verdict,"clf_verdict":clf_verdict,
        "confidence":confidence,"ensemble_agreement":agree,
        "star_power_used":star_val,"dir_power_used":dir_val,
    })
    print("  ✓ Saved to database")

except KeyboardInterrupt:
    print("\n\n  Prediction skipped.")

print("\n✓ Done.\n")