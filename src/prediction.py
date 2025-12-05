import numpy as np
import pandas as pd
import json
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from tensorflow.keras.models import load_model


def evaluate_predictions(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return acc, prec, rec, f1


csv_path = "../data/nifty50_ticks.csv"
df = pd.read_csv(csv_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
df = df[["timestamp", "open", "high", "low", "close"]]

df["next_close"] = df["close"].shift(-1)
df["fwd_ret"] = df["next_close"] / df["close"] - 1
df["prev_close"] = df["close"].shift(1)
df["body"] = df["close"] - df["open"]
df["upper_wick"] = df["high"] - df[["close", "open"]].max(axis=1)
df["lower_wick"] = df[["close", "open"]].min(axis=1) - df["low"]
df["range"] = df["high"] - df["low"]
df["return_1"] = df["close"] / df["prev_close"] - 1
df["return_3"] = df["close"].pct_change(3)
df["range_pct"] = df["range"] / df["close"]
df["body_pct"] = df["body"] / df["range"].replace(0, 1e-6)
df["day"] = df["timestamp"].dt.dayofweek

df = df.dropna().reset_index(drop=True)

with open("classical_meta.json", "r") as f:
    meta = json.load(f)

feature_cols = meta["feature_cols"]
train_ratio = meta["train_ratio"]
up_thresh = meta["up_thresh"]
is_tf_model = meta["is_tf_model"]
best_model_name = meta["best_model_name"]

X = df[feature_cols].values
y = (df["fwd_ret"] > up_thresh).astype(int).values

n = len(X)
train_size = int(train_ratio * n)

X_test = X[train_size:]
y_test = y[train_size:]
close_test = df["close"].values[train_size:]
next_close_test = df["next_close"].values[train_size:]

scaler = joblib.load("classical_scaler.pkl")
X_test_scaled = scaler.transform(X_test)

if is_tf_model:
    model = load_model("best_tf_mlp.keras")
    y_proba = model.predict(X_test_scaled).ravel()
else:
    model = joblib.load("best_classical_model.pkl")
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test_scaled)
        y_proba = 1 / (1 + np.exp(-scores))
    else:
        y_raw = model.predict(X_test_scaled)
        y_proba = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min() + 1e-8)

best_f1 = -1.0
best_threshold = 0.5
best_scores = (0.0, 0.0, 0.0, 0.0)
best_preds = None

print("\nBest model from training:", best_model_name)
print("\nSearching best probability threshold for F1...\n")

for thresh in np.linspace(0.05, 0.95, 50):
    y_pred = (y_proba > thresh).astype(int)
    acc, prec, rec, f1 = evaluate_predictions(y_test, y_pred)

    print(
        f"Thresh={thresh:.3f} -> Acc={acc:.3f} "
        f"Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh
        best_scores = (acc, prec, rec, f1)
        best_preds = y_pred

print("\n=== BEST THRESHOLD RESULTS ===")
acc, prec, rec, f1 = best_scores
print(f"Best Thresh: {best_threshold:.3f}")
print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1 Score: {f1:.3f}")

cm = confusion_matrix(y_test, best_preds)
print("\nConfusion Matrix:\n", cm)
print(
    "\nClassification Report:\n",
    classification_report(y_test, best_preds, zero_division=0),
)

pnl = ((next_close_test - close_test) / close_test) * best_preds
cum_pnl = pnl.cumsum()

df_out = pd.DataFrame(
    {
        "timestamp": df["timestamp"].values[train_size:],
        "close": close_test,
        "next_close": next_close_test,
        "pred": best_preds,
        "pnl": pnl,
        "cumulative_pnl": cum_pnl,
    }
)

df_out.to_csv("final_trading_results.csv", index=False)

print("\nTotal PnL:", float(cum_pnl[-1]) if len(cum_pnl) > 0 else 0.0)
print("Results saved to final_trading_results.csv")
