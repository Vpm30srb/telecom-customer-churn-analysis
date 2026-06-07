"""
Telecom Customer Churn Prediction
Step 5: Prediction Pipeline — Predict on new customer data
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_artifacts(model_dir='models'):
    """Load trained model and scaler."""
    try:
        model  = joblib.load(f'{model_dir}/XGBoost_tuned.pkl')
        print("Loaded: XGBoost_tuned.pkl")
    except FileNotFoundError:
        model  = joblib.load(f'{model_dir}/XGBoost.pkl')
        print("Loaded: XGBoost.pkl")
    scaler = joblib.load(f'{model_dir}/scaler.pkl')
    return model, scaler


def preprocess_input(df_raw, reference_columns_path='models/X_train.csv'):
    """Replicate the full preprocessing on new/unseen data."""
    df = df_raw.copy()

    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)
    if 'Churn' in df.columns:
        df.drop(columns=['Churn'], inplace=True)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)

    # Feature engineering
    df['AvgMonthlyCharge'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['NumServices'] = (
        (df['PhoneService'] == 'Yes').astype(int) +
        (df.get('MultipleLines', pd.Series(['No']*len(df))) == 'Yes').astype(int) +
        (df['InternetService'] != 'No').astype(int) +
        (df['OnlineSecurity'] == 'Yes').astype(int) +
        (df['OnlineBackup'] == 'Yes').astype(int) +
        (df['DeviceProtection'] == 'Yes').astype(int) +
        (df['TechSupport'] == 'Yes').astype(int) +
        (df['StreamingTV'] == 'Yes').astype(int) +
        (df['StreamingMovies'] == 'Yes').astype(int)
    )
    df['ChargePerService'] = df['MonthlyCharges'] / (df['NumServices'] + 1)
    df['TenureGroup'] = pd.cut(df['tenure'],
                                bins=[0, 12, 24, 48, 72],
                                labels=['0-1yr', '1-2yr', '2-4yr', '4+yr'])
    df['IsAutoPayment'] = df['PaymentMethod'].isin(
        ['Bank transfer (automatic)', 'Credit card (automatic)']).astype(int)
    df['HasSecurityServices'] = (
        (df['OnlineSecurity'] == 'Yes') | (df['TechSupport'] == 'Yes')
    ).astype(int)

    # Encoding
    binary_map = {'Male': 1, 'Female': 0, 'Yes': 1, 'No': 0}
    for col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        if col in df.columns:
            df[col] = df[col].map(binary_map)

    ohe_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract',
                'PaymentMethod', 'TenureGroup']
    df = pd.get_dummies(df, columns=[c for c in ohe_cols if c in df.columns], drop_first=True)

    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Align columns with training set
    ref_cols = pd.read_csv(reference_columns_path).columns.tolist()
    for col in ref_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[ref_cols]

    return df


def predict_churn(df_raw, model_dir='models', threshold=0.5):
    """Full prediction pipeline — returns df with churn probability and label."""
    model, scaler = load_artifacts(model_dir)
    df_proc = preprocess_input(df_raw, f'{model_dir}/X_train.csv')

    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges',
                'AvgMonthlyCharge', 'ChargePerService', 'NumServices']
    num_cols = [c for c in num_cols if c in df_proc.columns]
    df_proc[num_cols] = scaler.transform(df_proc[num_cols])

    probs  = model.predict_proba(df_proc)[:, 1]
    labels = (probs >= threshold).astype(int)

    result = df_raw.copy()
    if 'customerID' not in result.columns:
        result.insert(0, 'customerID', [f'NEW-{i+1:04d}' for i in range(len(result))])
    result['ChurnProbability'] = np.round(probs * 100, 2)
    result['ChurnPrediction']  = labels
    result['RiskLevel'] = pd.cut(
        probs,
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low', 'Medium', 'High']
    )

    return result[['customerID', 'ChurnProbability', 'ChurnPrediction', 'RiskLevel']]


# ── Demo ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    sample = pd.DataFrame([{
        'customerID': 'TEST-001',
        'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No',
        'Dependents': 'No', 'tenure': 3,
        'PhoneService': 'Yes', 'MultipleLines': 'No',
        'InternetService': 'Fiber optic', 'OnlineSecurity': 'No',
        'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No',
        'StreamingTV': 'Yes', 'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 95.45, 'TotalCharges': 286.35
    }, {
        'customerID': 'TEST-002',
        'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes',
        'Dependents': 'Yes', 'tenure': 60,
        'PhoneService': 'Yes', 'MultipleLines': 'Yes',
        'InternetService': 'DSL', 'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes', 'DeviceProtection': 'Yes', 'TechSupport': 'Yes',
        'StreamingTV': 'No', 'StreamingMovies': 'No',
        'Contract': 'Two year', 'PaperlessBilling': 'No',
        'PaymentMethod': 'Credit card (automatic)',
        'MonthlyCharges': 55.20, 'TotalCharges': 3312.00
    }])

    result = predict_churn(sample)
    print("\nPrediction Results:")
    print(result.to_string(index=False))
