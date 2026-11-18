# Overnight Water Shortage Prediction – End-to-End MLOps

## Project overview
This project demonstrates a simple end-to-end MLOps workflow that predicts whether a residential water tank may fall below a critical reserve level by 6:00 AM.

The synthetic dataset represents a 10-hour overnight period from 8:00 PM to 6:00 AM. A shortage is labelled when the projected morning tank level is below 15% of the 100,000-litre tank capacity (15,000 litres).

> **Academic demonstration:** The datasets are synthetic and generated using assumed consumption, pumping, and leakage ranges. Results are not validated against real residential water-system measurements.

## Problem statement
A residential colony receives municipal water through an overhead tank. Water is consumed overnight, and the pump operates only for scheduled periods. The objective is to predict shortage risk using the 8:00 PM tank level, number of households, consumption rate, pumping details, leakage estimate, and day type.

## Technology stack
- Python
- NumPy and Pandas
- Scikit-learn
- MLflow
- FastAPI and Uvicorn
- Google Colab / Jupyter Notebook

## MLOps workflow
1. **Dataset creation and versioning:** Generate V1 normal-condition data and V2 summer-condition data (35% higher assumed consumption).
2. **Preprocessing and feature engineering:** Calculate expected overnight demand, expected pump supply, and projected water buffer.
3. **Model training:** Train Decision Tree and Random Forest classifiers.
4. **Evaluation and selection:** Compare accuracy, precision, recall, and F1-score; select the model with the higher F1-score.
5. **Experiment tracking:** Log model information and metrics with MLflow.
6. **CI/CD-style validation:** Run basic assertions on tank capacity, input values, and prediction output.
7. **Model serving:** Expose a FastAPI `/predict` endpoint.

## Repository structure
```text
overnight-water-shortage-mlops/
├── data/
│   ├── water_tank_v1.csv
│   ├── water_tank_v2.csv
│   └── feature_store.csv
├── models/
│   └── water_shortage_model.joblib
├── results/
│   ├── evaluation_metrics.json
│   └── confusion_matrix.png
├── Overnight_Water_Shortage_MLOps.ipynb
├── pipeline.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset fields
| Feature | Description |
|---|---|
| `tank_level_8pm` | Water available at 8:00 PM (litres) |
| `households` | Number of connected households |
| `avg_hourly_consumption` | Assumed average litres consumed per household per hour |
| `pumping_duration` | Scheduled pump duration (hours) |
| `pumping_rate` | Pump supply rate (litres/hour) |
| `leakage_rate` | Estimated leakage (litres/hour) |
| `day_type` | 0 = weekend, 1 = weekday |
| `morning_water_shortage` | Target: 1 = shortage, 0 = adequate reserve |

## Evaluation snapshot
The generated test-set metrics are stored in [`results/evaluation_metrics.json`](results/evaluation_metrics.json). The selected model is **Decision Tree**.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

See [`results/confusion_matrix.png`](results/confusion_matrix.png) for the selected model's confusion matrix.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python pipeline.py
uvicorn app:app --reload
```
Open `http://127.0.0.1:8000/docs` to test the API interactively.

## Example API request
Send a POST request to `/predict` with:
```json
{
  "tank_level_8pm": 40000,
  "households": 120,
  "avg_hourly_consumption": 25,
  "pumping_duration": 1.5,
  "pumping_rate": 9500,
  "leakage_rate": 250,
  "day_type": 1
}
```

## Important limitations
- Data is simulated, not collected from physical tank sensors.
- Model scores measure performance on a small synthetic test split only.
- The feature calculations assume a fixed 10-hour overnight period.
- This project demonstrates experiment tracking and API serving; it does not include an automated cloud deployment or a production CI/CD server.
