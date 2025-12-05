# TRADIO — Next-Candle Direction Prediction (NIFTY50)

This project applies Machine Learning to predict whether the "next intraday candle" will close higher than the current one (by using a minimal gain threshold). Predictions are used to simulate a simple trading strategy and evaluate overall profitability for the users using the model.

---



TRADIO_PROJECT/
├── data/                
├── src/
│   ├── model.py         
│   └── prediction.py    
├── results/
│   └── final_trading_results.csv
├── requirements.txt
└── README.md

---



1) Load and clean intraday OHLC data  
2) Generate price-action features (body, wicks, momentum, volatility)  
3) Train multiple classical ML models + XGBoost + TF-MLP  
4) Select best model based on **F1 score**  
5) Perform probability threshold sweep  
6) Generate trading signals → calculate PnL

---

 Best Model & Results

XGBoost Classifier (evaluated on unseen test data)

- F1 Score ≈ **0.40**
- Positive Trading PnL ✔
- Significant improvement over random baseline

> Even a small predictive edge can be profitable when applied consistently in intraday trading 📈

---

#How to Run

```bash
pip install -r requirements.txt

cd src
python model.py        
python prediction.py  
