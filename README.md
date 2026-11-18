
# Overnight Water Shortage Prediction – End-to-End MLOps Project

## Student Details

**Name:** SHomya raj

**Registration Number:** 231FA04A51

**Section:** **SEC 15**

---

## Project Description

This project focuses on predicting overnight water shortages in residential water tanks using Machine Learning and MLOps techniques.

The system analyzes tank water levels, household consumption, pumping duration, pumping rate and leakage estimates to predict whether sufficient water will be available by the next morning.

## Problem Statement

A residential colony receives municipal water through an overhead storage tank. Water is consumed throughout the night, and the pump operates only during scheduled hours.

The objective is to predict potential water shortages before they occur so that necessary actions can be taken.

## Technology Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- MLflow
- FastAPI
- Google Colab

## Project Workflow

1. Dataset Creation and Versioning
2. Data Preprocessing
3. Feature Engineering
4. Machine Learning Model Training
5. Model Evaluation
6. MLflow Experiment Tracking
7. CI/CD Testing
8. FastAPI Model Deployment
9. Water Shortage Prediction

## Machine Learning Models

- Decision Tree Classifier
- Random Forest Classifier

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## MLOps Implementation

| MLOps Component | Tool Used |
|---|---|
| Data Processing | Pandas |
| Feature Engineering | Python |
| Model Training | Scikit-learn |
| Experiment Tracking | MLflow |
| Model Validation | Automated Testing |
| Model Serving | FastAPI |
| Development Environment | Google Colab |

## Dataset Information

Two synthetic datasets are generated:

- **Dataset V1:** Normal water consumption conditions.
- **Dataset V2:** Summer conditions with increased water consumption.

The target variable is `morning_water_shortage`.

- 1 = Shortage Risk
- 0 = Adequate Reserve

## Expected Outcome

The system predicts the possibility of overnight water shortages based on available tank information and estimated water consumption.

It helps demonstrate how Machine Learning and MLOps can be used to develop, evaluate, track and serve a predictive model.

## Conclusion

This project demonstrates an end-to-end MLOps pipeline for overnight water shortage prediction. It combines data processing, machine learning, experiment tracking, automated testing and API deployment into a single workflow.

---

**Developed By: SHomya raj**

**Registration Number: 231FA04A51**

**SECTION: SEC 15**
