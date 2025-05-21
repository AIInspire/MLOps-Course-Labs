# 🧠 Churn Prediction Experiment

This project focuses on building and comparing machine learning models to predict customer churn using performance metrics tracked and visualized in **MLflow**.

---

## 📊 Experiment Summary

- **Platform**: MLflow 2.22.0  
- **Task**: Binary classification (churn vs. no churn)  
- **Metrics Tracked**:  
  - Accuracy  
  - F1 Score  
  - Precision  
  - Recall

---

## 🏆 Best Model: `xgboost`

### 🔍 Reason for Selection

After comparing two models (`xgboost` vs. `random_for`) with similar hyperparameters (`max_depth` ~ 5), the `xgboost` model consistently outperformed `random_for` in **3 out of 4 metrics**:

| Metric      | XGBoost    | Random Forest | Winner       |
|-------------|------------|----------------|--------------|
| Accuracy    | 0.7637     | 0.7621         | ✅ XGBoost   |
| F1 Score    | 0.7529     | 0.7519         | ✅ XGBoost   |
| Precision   | 0.7709     | 0.7689         | ✅ XGBoost   |
| Recall      | 0.7340     | 0.7421         | ❌ Random Forest |

Although Random Forest had a slightly higher recall, **XGBoost demonstrated a better balance** across all major classification metrics, making it the optimal choice for this project.

---

## 🛠 Parameters of Best Model

- **Model Type**: XGBoost  
- **Max Depth**: 5  
- *(Other hyperparameters can be added if available)*

---

## 🚀 How to Run

1. Clone this repository  
2. Install dependencies from `requirements.txt`  
3. Train models using your training script  
4. Launch MLflow UI to view and compare experiments:
   ```bash
   mlflow ui
