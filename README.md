# 🎬 Indian Box Office Predictor

Predicts lifetime worldwide collection and verdict (HIT / FLOP / BLOCKBUSTER etc.)
for Indian films using XGBoost — trained on 500+ films from 2021–2024.

---

## 📁 Project Structure

```
box_office_predictor/
│
├── data/                          ← Put all 4 CSV files here
│   ├── Indian_Movies_2021_updated.csv
│   ├── Indian_Movies_2022_updated.csv
│   ├── Indian_Movies_2023_updated.csv
│   └── Indian_Movies_2024_updated.csv
│
├── models/                        ← Auto-created after training
│   ├── regressor.pkl
│   ├── classifier.pkl
│   ├── label_encoder.pkl
│   ├── label_language.pkl
│   ├── label_season.pkl
│   ├── star_power_map.pkl
│   ├── director_power_map.pkl
│   ├── meta.pkl
│   └── results.png
│
├── main.py                        ← Train the models
├── app.py                         ← Streamlit web app
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup (one time)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your CSV files
Put all 4 updated CSV files inside a `data/` folder.

### 3. Train the models
```bash
python main.py
```
This will:
- Load and clean all 4 CSV files
- Train XGBoost regressor + classifier
- Print R2 score and accuracy
- Save all models to `models/`
- Show and save result charts

**Good scores to expect:**
- R2 Score: above 0.75
- Accuracy: above 55%

### 4. Run the web app
```bash
streamlit run app.py
```
Opens a browser at http://localhost:8501

---

## 🌐 Deploy Free on Streamlit Cloud

1. Push this entire folder to a GitHub repo
2. Go to https://streamlit.io/cloud
3. Click "New app" → select your repo → set main file as `app.py`
4. Done — you get a public shareable link!

---

## 🔮 How Predictions Work

**Features us
- Budget (₹ Crore)
- Opening Day collection (₹ Crore)
- Number of screens
- Lead star's historical median worldwide
- Director's historical average worldwide
- Language
- Release season (Holiday / Summer / Monsoon / Normal)
- Franchise / sequel flag
- Release year
- Derived ratios (opening/budget, screens/budget, opening/screen)

**Two models:**
- XGBoost Regressor → predicts worldwide collection in ₹ Crore
- XGBoost Classifier → predicts verdict (DISASTER / FLOP / AVERAGE / HIT / SUPER HIT / BLOCKBUSTER / ALL TIME BLOCKBUSTER)