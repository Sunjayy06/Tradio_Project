import numpy as np
import pandas as pd
import json
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    f1_score,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

csv_path = "../data/nifty50_ticks.csv"
up_thresh = 0.00015
train_ratio = 0.7

np.random.seed(42)
tf.random.set_seed(42)

df = pd.read_csv(csv_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
df = df[["timestamp", "open", "high", "low", "close"]]

df["next_close"] = df["close"].shift(-1)
df["fwd_ret"] = df["next_close"] / df["close"] - 1
df["target"] = (df["fwd_ret"] > up_thresh).astype(int)

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

feature_cols = [
    "open", "high", "low", "close",
    "body", "upper_wick", "lower_wick", "range",
    "return_1", "return_3",
    "range_pct", "body_pct",
    "day",
]

X = df[feature_cols].values
y = df["target"].values

n = len(X)
train_size = int(train_ratio * n)

X_train_full = X[:train_size]
X_test = X[train_size:]
y_train_full = y[:train_size]
y_test = y[train_size:]

inner_train_ratio = 0.8
inner_train_size = int(inner_train_ratio * train_size)

X_inner_train = X_train_full[:inner_train_size]
y_inner_train = y_train_full[:inner_train_size]
X_val = X_train_full[inner_train_size:]
y_val = y_train_full[inner_train_size:]

print("Models (classical + boosting + TF_MLP):")
print("Total samples:", n)
print("Train size   :", train_size)
print("Inner train  :", inner_train_size)
print("Val size     :", train_size - inner_train_size)
print("Test size    :", n - train_size)

print("\nTrain class 1 ratio:", y_train_full.mean())
print("Test  class 1 ratio:", y_test.mean())

scaler = StandardScaler()
scaler.fit(X_inner_train)

X_inner_train_scaled = scaler.transform(X_inner_train)
X_val_scaled = scaler.transform(X_val)
X_train_scaled = scaler.transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

results = {}
best_model_name_global = None
best_score_global = -1.0
best_model_global = None
best_is_tf = False

print("\nHyperparameter search per model (using validation set):")

logreg_configs = [{"C": 0.1}, {"C": 0.5}, {"C": 1.0}]
dt_configs = [
    {"max_depth": 5, "min_samples_leaf": 10},
    {"max_depth": 8, "min_samples_leaf": 10},
    {"max_depth": 12, "min_samples_leaf": 15},
]
svm_configs = [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}]
rf_configs = [
    {"n_estimators": 150, "max_depth": 6, "min_samples_leaf": 10},
    {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 10},
    {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 15},
]
gb_configs = [
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05},
    {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.03},
]
xgb_configs = [
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.8},
    {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.8},
    {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.8},
]
tf_mlp_configs = [{"name": "tf_mlp"}]

model_grids = {
    "LogisticRegression": logreg_configs,
    "DecisionTree": dt_configs,
    "SVM": svm_configs,
    "RandomForest": rf_configs,
    "GradientBoosting": gb_configs,
    "XGBoost": xgb_configs,
    "TF_MLP": tf_mlp_configs,
}

pos_ratio = y_inner_train.mean()
neg_ratio = 1.0 - pos_ratio
if pos_ratio > 0:
    scale_pos_weight = neg_ratio / pos_ratio
else:
    scale_pos_weight = 1.0
print("\nInner train class 1 ratio:", pos_ratio, "-> scale_pos_weight for XGBoost:", scale_pos_weight)

for model_name, config_list in model_grids.items():
    print(f"\n {model_name }\n")
    best_val_f1 = -1.0
    best_config = None

    if model_name != "TF_MLP":
        for cfg in config_list:
            if model_name == "LogisticRegression":
                model = LogisticRegression(
                    C=cfg["C"],
                    penalty="l2",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                )
            elif model_name == "DecisionTree":
                model = DecisionTreeClassifier(
                    max_depth=cfg["max_depth"],
                    min_samples_leaf=cfg["min_samples_leaf"],
                    class_weight="balanced",
                    random_state=42,
                )
            elif model_name == "SVM":
                model = LinearSVC(
                    C=cfg["C"],
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=42,
                )
            elif model_name == "RandomForest":
                model = RandomForestClassifier(
                    n_estimators=cfg["n_estimators"],
                    max_depth=cfg["max_depth"],
                    min_samples_leaf=cfg["min_samples_leaf"],
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=42,
                )
            elif model_name == "GradientBoosting":
                model = GradientBoostingClassifier(
                    n_estimators=cfg["n_estimators"],
                    learning_rate=cfg["learning_rate"],
                    max_depth=cfg["max_depth"],
                    random_state=42,
                )
            elif model_name == "XGBoost":
                model = XGBClassifier(
                    n_estimators=cfg["n_estimators"],
                    max_depth=cfg["max_depth"],
                    learning_rate=cfg["learning_rate"],
                    subsample=cfg["subsample"],
                    colsample_bytree=cfg["colsample_bytree"],
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos_weight,
                    n_jobs=-1,
                    tree_method="hist",
                    random_state=42,
                    verbosity=0,
                )
            else:
                continue

            model.fit(X_inner_train_scaled, y_inner_train)
            y_val_pred = model.predict(X_val_scaled)
            val_f1 = f1_score(y_val, y_val_pred, zero_division=0)

            print(f"{model_name} config {cfg} -> val_f1 = {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_config = cfg

        print(f"Best {model_name} config: {best_config} with val_f1 = {best_val_f1:.4f}")

    else:
        class_weights_arr = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1]),
            y=y_inner_train,
        )
        class_weight_dict = {0: class_weights_arr[0], 1: class_weights_arr[1]}
        input_dim = X_inner_train_scaled.shape[1]

        model = models.Sequential(
            [
                layers.Input(shape=(input_dim,)),
                layers.Dense(64, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

        optimizer = optimizers.Adam(learning_rate=0.001)

        model.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"],
        )

        es = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )

        model.fit(
            X_inner_train_scaled,
            y_inner_train,
            validation_data=(X_val_scaled, y_val),
            epochs=15,
            batch_size=256,
            class_weight=class_weight_dict,
            verbose=0,
            callbacks=[es],
        )

        y_val_proba = model.predict(X_val_scaled).ravel()
        y_val_pred = (y_val_proba > 0.5).astype(int)
        val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
        print(f"TF_MLP -> val_f1 = {val_f1:.4f}")
        best_val_f1 = val_f1
        best_config = {"tf": True}

    if model_name == "LogisticRegression":
        final_model = LogisticRegression(
            C=best_config["C"],
            penalty="l2",
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )
        final_model.fit(X_train_scaled, y_train_full)
        y_test_pred = final_model.predict(X_test_scaled)

    elif model_name == "DecisionTree":
        final_model = DecisionTreeClassifier(
            max_depth=best_config["max_depth"],
            min_samples_leaf=best_config["min_samples_leaf"],
            class_weight="balanced",
            random_state=42,
        )
        final_model.fit(X_train_scaled, y_train_full)
        y_test_pred = final_model.predict(X_test_scaled)

    elif model_name == "SVM":
        final_model = LinearSVC(
            C=best_config["C"],
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        )
        final_model.fit(X_train_scaled, y_train_full)
        y_test_pred = final_model.predict(X_test_scaled)

    elif model_name == "RandomForest":
        final_model = RandomForestClassifier(
            n_estimators=best_config["n_estimators"],
            max_depth=best_config["max_depth"],
            min_samples_leaf=best_config["min_samples_leaf"],
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        final_model.fit(X_train_scaled, y_train_full)
        y_test_pred = final_model.predict(X_test_scaled)

    elif model_name == "GradientBoosting":
        final_model = GradientBoostingClassifier(
            n_estimators=best_config["n_estimators"],
            learning_rate=best_config["learning_rate"],
            max_depth=best_config["max_depth"],
            random_state=42,
        )
        final_model.fit(X_train_scaled, y_train_full)
        y_test_pred = final_model.predict(X_test_scaled)

    elif model_name == "XGBoost":
        final_model = XGBClassifier(
            n_estimators=best_config["n_estimators"],
            max_depth=best_config["max_depth"],
            learning_rate=best_config["learning_rate"],
            subsample=best_config["subsample"],
            colsample_bytree=best_config["colsample_bytree"],
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            tree_method="hist",
            random_state=42,
            verbosity=0,
        )
        final_model.fit(X_train_scaled, y_train_full)
        y_test_pred = final_model.predict(X_test_scaled)

    elif model_name == "TF_MLP":
        class_weights_arr = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1]),
            y=y_train_full,
        )
        class_weight_dict = {0: class_weights_arr[0], 1: class_weights_arr[1]}
        input_dim = X_train_scaled.shape[1]

        final_model = models.Sequential(
            [
                layers.Input(shape=(input_dim,)),
                layers.Dense(64, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

        final_model.compile(
            loss="binary_crossentropy",
            optimizer=optimizers.Adam(learning_rate=0.001),
            metrics=["accuracy"],
        )

        final_model.fit(
            X_train_scaled,
            y_train_full,
            validation_split=0.2,
            epochs=15,
            batch_size=256,
            class_weight=class_weight_dict,
            verbose=0,
        )

        y_test_proba = final_model.predict(X_test_scaled).ravel()
        y_test_pred = (y_test_proba > 0.5).astype(int)

    else:
        continue

    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred, zero_division=0)
    test_rec = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_test_pred)

    results[model_name] = {
        "accuracy": test_acc,
        "precision": test_prec,
        "recall": test_rec,
        "f1": test_f1,
    }

    print(f"\n {model_name} ")
    print("Accuracy :", test_acc)
    print("Precision:", test_prec)
    print("Recall   :", test_rec)
    print("F1 score :", test_f1)
    print("\nConfusion matrix:\n", cm)
    print(
        "\nClassification report:\n",
        classification_report(y_test, y_test_pred, zero_division=0),
    )

    if test_f1 > best_score_global:
        best_score_global = test_f1
        best_model_name_global = model_name
        best_model_global = final_model
        best_is_tf = (model_name == "TF_MLP")

print("\n*** Best model (by F1):", best_model_name_global, "with test F1:", best_score_global, "***")

if best_is_tf:
    best_model_global.save("best_tf_mlp.keras")
else:
    joblib.dump(best_model_global, "best_classical_model.pkl")

joblib.dump(scaler, "classical_scaler.pkl")

meta = {
    "best_model_name": best_model_name_global,
    "train_ratio": train_ratio,
    "feature_cols": feature_cols,
    "up_thresh": up_thresh,
    "is_tf_model": best_is_tf,
}
with open("classical_meta.json", "w") as f:
    json.dump(meta, f)

print("Saved best model and scaler.")
