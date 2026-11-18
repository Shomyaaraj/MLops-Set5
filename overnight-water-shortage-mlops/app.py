import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Overnight Water Shortage Prediction API")
model = joblib.load("models/water_shortage_model.joblib")

FEATURES = [
    "tank_level_8pm", "households", "avg_hourly_consumption",
    "pumping_duration", "pumping_rate", "leakage_rate", "day_type",
    "expected_overnight_demand", "expected_pump_supply", "projected_water_buffer"
]

class TankData(BaseModel):
    tank_level_8pm: float = Field(ge=0, le=100000)
    households: int = Field(gt=0)
    avg_hourly_consumption: float = Field(ge=0)
    pumping_duration: float = Field(ge=0, le=10)
    pumping_rate: float = Field(ge=0)
    leakage_rate: float = Field(ge=0)
    day_type: int = Field(ge=0, le=1)

@app.get("/")
def home():
    return {"message": "Water Shortage Prediction API is running"}

@app.post("/predict")
def predict(data: TankData):
    demand = data.households * data.avg_hourly_consumption * 10
    pumping = data.pumping_duration * data.pumping_rate
    buffer = data.tank_level_8pm - demand - data.leakage_rate*10 + pumping
    row = pd.DataFrame([{
        "tank_level_8pm": data.tank_level_8pm,
        "households": data.households,
        "avg_hourly_consumption": data.avg_hourly_consumption,
        "pumping_duration": data.pumping_duration,
        "pumping_rate": data.pumping_rate,
        "leakage_rate": data.leakage_rate,
        "day_type": data.day_type,
        "expected_overnight_demand": demand,
        "expected_pump_supply": pumping,
        "projected_water_buffer": buffer
    }])[FEATURES]
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])
    return {
        "Prediction": "Shortage Risk" if prediction == 1 else "Adequate Reserve",
        "Shortage_Probability": round(probability, 4),
        "Projected_Water_Buffer": round(float(buffer), 2)
    }
