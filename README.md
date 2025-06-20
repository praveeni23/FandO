# 📈 Nifty 50 Market Movement Prediction (Classification Model)

This project applies machine learning and deep learning models to predict the **next-day direction** (Up or Down) of the **Nifty 50 index** using 10+ years of historical market data and technical indicators.

---

## 🧠 Objective

To build a predictive model that classifies whether the NIFTY 50 index will move **up (1)** or **down (0)** the following day, based on historical price patterns and technical indicators.

---

## ⚙️ Technologies Used

- 🐍 Python (Pandas, NumPy)
- 📊 yFinance (for data extraction)
- 📈 Technical Analysis (RSI, MACD, Bollinger Bands, ATR)
- 🤖 ML Models: 
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - Logistic Regression
- 🧠 Deep Learning: Keras Sequential Model
- 📉 Scikit-learn: Preprocessing, Ensemble Voting, Evaluation

---

## 📌 Features Engineered

- Moving Averages (MA10, MA50)
- RSI (Relative Strength Index)
- Bollinger Bands (Upper & Lower)
- MACD & MACD Signal
- ATR (Average True Range)
- Daily Returns

---

## 📊 Results

- **Model Type:** Ensemble Voting Classifier  
- **Accuracy Achieved:** ~55% (baseline model, tunable)
- **Metrics Used:** Accuracy, F1-score, Precision, Recall  
- **Final Output:** Close price, actual direction, predicted direction

---

## 📁 Folder Structure

nifty50-prediction/
├── data/ # optional
├── src/
│ └── nifty50_prediction.py
├── README.md
└── requirements.txt # (create with pip freeze > requirements.txt)

yaml
Copy code

---

## 📈 Sample Output

| Date       | Close Price | Target | Prediction |
|------------|-------------|--------|------------|
| 2025-06-16 | 24946.50    | 0      | 1          |
| 2025-06-17 | 24853.40    | 0      | 1          |
| 2025-06-18 | 24812.05    | 0      | 1          |
| 2025-06-19 | 24793.25    | 1      | 1          |
| 2025-06-20 | 25112.40    | 0      | 1          |

---

## ✅ Future Work

- Integrate LSTM or Transformer-based deep learning model
- Use options data to predict strike price zone
- Build a backtesting pipeline for real-world validation
- Tune threshold to maximize precision or minimize false positives

---

## 🧑‍💼 Author

**Praveen I.**  
HR Analyst & Aspiring Data Scientist  
