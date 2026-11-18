import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
np.random.seed(42)
N, CAPACITY = 250, 100000

def generate_dataset(version="v1"):
    level = np.random.uniform(20000, CAPACITY, N)
    homes = np.random.randint(50, 150, N)
    factor = 1.35 if version == "v2" else 1.0
    use = np.random.uniform(15, 30, N) * factor
    pump_hours = np.random.uniform(0, 4, N)
    pump_rate = np.random.uniform(8000, 12000, N)
    leak = np.random.uniform(100, 500, N)
    day = np.random.randint(0, 2, N)
    morning = np.minimum(level - homes*use*10 + pump_hours*pump_rate - leak*10, CAPACITY)
    return pd.DataFrame({
        "tank_level_8pm": level, "households": homes,
        "avg_hourly_consumption": use, "pumping_duration": pump_hours,
        "pumping_rate": pump_rate, "leakage_rate": leak, "day_type": day,
        "morning_water_shortage": (morning < 15000).astype(int)
    })

def engineer(df):
    df = df.copy()
    df["expected_overnight_demand"] = df["households"] * df["avg_hourly_consumption"] * 10
    df["expected_pump_supply"] = df["pumping_duration"] * df["pumping_rate"]
    df["projected_water_buffer"] = (
        df["tank_level_8pm"] - df["expected_overnight_demand"]
        - df["leakage_rate"]*10 + df["expected_pump_supply"]
    )
    return df

if __name__ == "__main__":
    v1, v2 = generate_dataset("v1"), generate_dataset("v2")
    v1.to_csv("data/water_tank_v1.csv", index=False)
    v2.to_csv("data/water_tank_v2.csv", index=False)
    data = engineer(v2)
    data.to_csv("data/feature_store.csv", index=False)
    features = [
        "tank_level_8pm", "households", "avg_hourly_consumption",
        "pumping_duration", "pumping_rate", "leakage_rate", "day_type",
        "expected_overnight_demand", "expected_pump_supply", "projected_water_buffer"
    ]
    X, y = data[features], data["morning_water_shortage"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    candidates = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42)
    }
    scores = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        scores[name] = f1_score(y_test, pred, zero_division=0)
    best_name = max(scores, key=scores.get)
    best_model = candidates[best_name]
    joblib.dump(best_model, "models/water_shortage_model.joblib")
    mlflow.set_experiment("Water_Shortage_Prediction")
    with mlflow.start_run():
        mlflow.log_param("selected_model", best_name)
        mlflow.log_param("dataset_version", "v2")
        mlflow.log_metric("f1_score", scores[best_name])
        mlflow.sklearn.log_model(
            best_model, name="water_shortage_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
    print("Selected model:", best_name)
    print("F1 scores:", scores)
