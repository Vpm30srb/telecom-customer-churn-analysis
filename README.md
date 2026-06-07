# 📡 Telecom Customer Churn Prediction

A complete end-to-end Machine Learning project to predict customer churn for a telecom company using Python, Scikit-learn, and XGBoost.

---

## 📌 Problem Statement

Customer churn is one of the biggest challenges in the telecom industry. Retaining an existing customer costs **5x less** than acquiring a new one. This project builds a predictive model to identify customers at high risk of churning — enabling proactive retention strategies.

---

## 📁 Project Structure

```
telecom_churn/
├── data/
│   ├── telecom_churn.csv          # Main dataset (7,043 customers, 21 features)
│   └── generate_dataset.py        # Script to regenerate the dataset
├── src/
│   ├── 01_eda.py                  # Exploratory Data Analysis
│   ├── 02_preprocessing.py        # Feature engineering & preprocessing
│   ├── 03_train_models.py         # Train 5 ML models + evaluation
│   ├── 04_hyperparameter_tuning.py # GridSearchCV tuning (XGBoost)
│   └── 05_predict.py              # Inference pipeline for new data
├── models/                        # Saved .pkl model files & processed data
├── reports/
│   └── figures/                   # All EDA & evaluation plots
├── main.py                        # Run full pipeline end-to-end
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset Overview

| Feature           | Type        | Description                          |
|-------------------|-------------|--------------------------------------|
| customerID        | String      | Unique customer identifier           |
| gender            | Categorical | Male / Female                        |
| SeniorCitizen     | Binary      | 1 = Senior, 0 = Not                 |
| Partner           | Binary      | Has partner                          |
| Dependents        | Binary      | Has dependents                       |
| tenure            | Numeric     | Months with the company              |
| PhoneService      | Binary      | Has phone service                    |
| MultipleLines     | Categorical | Single / Multiple / No phone         |
| InternetService   | Categorical | DSL / Fiber optic / No               |
| OnlineSecurity    | Binary      | Has online security add-on           |
| TechSupport       | Binary      | Has tech support                     |
| Contract          | Categorical | Month-to-month / 1yr / 2yr           |
| PaperlessBilling  | Binary      | Paperless billing                    |
| PaymentMethod     | Categorical | 4 payment methods                    |
| MonthlyCharges    | Numeric     | Monthly bill ($)                     |
| TotalCharges      | Numeric     | Total billed to date ($)             |
| **Churn**         | **Target**  | **Yes / No**                         |

**Dataset size:** 7,043 rows × 21 columns | **Churn rate:** ~33%

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/telecom-churn-prediction.git
cd telecom-churn-prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline
python main.py
```

---

## 🔬 ML Pipeline

```
Raw CSV  →  EDA  →  Feature Engineering  →  Preprocessing  →  SMOTE  →  Model Training  →  Evaluation  →  Prediction
```

### Feature Engineering
- `AvgMonthlyCharge` = TotalCharges / (tenure + 1)
- `NumServices` = count of all active services
- `ChargePerService` = MonthlyCharges / NumServices
- `TenureGroup` = binned tenure (0-1yr, 1-2yr, 2-4yr, 4+yr)
- `IsAutoPayment` = 1 if automatic payment method
- `HasSecurityServices` = 1 if OnlineSecurity or TechSupport active

### Models Trained
| Model               | Notes                               |
|---------------------|-------------------------------------|
| Logistic Regression | Baseline, class_weight='balanced'   |
| Random Forest       | 200 trees, max_depth=8              |
| XGBoost             | Gradient boosting, tuned            |
| Gradient Boosting   | sklearn GBM                         |
| KNN                 | k=9, euclidean distance             |

### Class Imbalance
Handled with **SMOTE** (Synthetic Minority Oversampling Technique) on the training set.

---

## 📈 Results

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~79%     | ~76%      | ~74%   | ~75%     | ~83%    |
| Random Forest       | ~81%     | ~78%      | ~76%   | ~77%     | ~85%    |
| **XGBoost**         | **~83%** | **~80%**  | **~79%**| **~79%**| **~87%** |
| Gradient Boosting   | ~82%     | ~79%      | ~78%   | ~78%     | ~86%    |
| KNN                 | ~76%     | ~73%      | ~71%   | ~72%     | ~80%    |

> ✅ **XGBoost** is the best performing model.

---

## 🔮 Predict on New Customers

```python
from src.predict import predict_churn
import pandas as pd

new_customer = pd.DataFrame([{
    'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No',
    'Dependents': 'No', 'tenure': 3,
    'PhoneService': 'Yes', 'MultipleLines': 'No',
    'InternetService': 'Fiber optic', 'OnlineSecurity': 'No',
    'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No',
    'StreamingTV': 'Yes', 'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 95.45, 'TotalCharges': 286.35
}])

result = predict_churn(new_customer)
print(result)
# Output: CustomerID | ChurnProbability | ChurnPrediction | RiskLevel
```

---

## 📊 Key Insights

- 📌 **Month-to-month contracts** have ~42% churn vs ~11% for 2-year contracts
- 📌 **Fiber optic** customers churn more — possibly due to pricing
- 📌 **New customers** (tenure < 6 months) are highest risk
- 📌 **Electronic check** payment method correlates with higher churn
- 📌 Customers **without TechSupport or OnlineSecurity** churn significantly more

---

## 🛠️ Technologies Used

- **Python 3.10+**
- **Pandas, NumPy** — data manipulation
- **Scikit-learn** — ML models, preprocessing, evaluation
- **XGBoost** — gradient boosting
- **Imbalanced-learn** — SMOTE oversampling
- **Matplotlib, Seaborn** — visualization
- **Joblib** — model serialization

---

## 👤 Author

**Vinayaka PM**
- GitHub: [@Vpm30srb](https://github.com/Vpm30srb)
- LinkedIn: [Vinayaka PM](https://www.linkedin.com/in/vinayakapm/)

---

## 📄 License

This project is licensed under the MIT License.
