# Customer Churn Prediction

ML model that predicts whether a telecom customer is likely to churn.
Built with Python, Scikit-learn, and Streamlit.

## Tech Stack
Python · Scikit-learn · Random Forest · Streamlit · Pandas · NumPy

## Model Selection
Trained and compared Logistic Regression, Random Forest, XGBoost, KNN, SVM, and Naive Bayes using GridSearchCV with 5-fold cross-validation.

| Model | ROC-AUC | F1-Score (Churn) | Recall (Churn) |
|---|---|---|---|
| Logistic Regression | 0.833 | 0.54 | 0.46 |
| XGBoost | 0.854 | 0.58 | 0.51 |
| Random Forest | 0.921| 0.68 | 0.61 |

Random Forest selected based on ROC-AUC and F1-Score on the minority churn class.

## Run Locally
```bash
git clone <repo-url>
cd customer-churn-prediction
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` — Dataset
- `app.py` — Streamlit application
- `col_w.pkl` — Feature columns reference
- `customer-churn-prediction.ipynb` — EDA, preprocessing, model training and comparison
- `customer_churn_model.pkl` — Trained Random Forest model
- `scaler.pkl` — StandardScaler fitted on tenure column



## Dataset
Telco Customer Churn — public dataset via Kaggle. 7043 rows, 21 features.

## Disclaimer
Educational project only. Not intended for production business decisions.