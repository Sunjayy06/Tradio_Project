TRADIO — Next-Candle Direction Prediction (NIFTY50)

This project applies Machine Learning to predict whether the next intraday candle will close higher than the current one based on a minimal gain threshold. Predictions are used to simulate a trading strategy and analyze potential profitability for better planning of users.

---

 Project Structure

```
TRADIO_PROJECT/
├── data/                
│   └── nifty50_ticks.csv
├── src/
│   ├── model.py           
│   └── prediction.py      
├── results/
│   └── final_trading_results.csv
├── requirements.txt
└── README.md
```

---

 Workflow

1) Load and process intraday OHLC data  
2) Engineer price-action features (body, range, returns, volatility)  
3) Train ML models: Logistic Regression, SVM, Decision Tree, Random Forest, Gradient Boosting  
4) Train XGBoost (best performing model)  
5) Probability threshold sweep for best trading signal quality  
6) Generate signals & compute PnL results  

---

Best Performing Model

**Model**: XGBoost Classifier  
Evaluated on unseen test data:

| Metric     | Score  |
|-----------|-------:|
| Accuracy  | ~0.44  |
| Precision | ~0.27  |
| Recall    | ~0.79  |
| **F1 Score** | **~0.40**  |

Positive simulated trading PnL  
Better than random baseline  
High recall = captures most upward moves  


 How to Run

Install dependencies:

bash
pip install -r requirements.txt


 Run training:
bash
'''
cd src
python model.py
python prediction.py
'''




