"""
Telecom Customer Churn Prediction
Step 2: Feature Engineering & Preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def load_and_clean(path='data/telecom_churn.csv'):
    df = pd.read_csv(path)

    # Drop customerID (not predictive)
    df.drop(columns=['customerID'], inplace=True)

    # TotalCharges: fill edge cases
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)

    # Binary target
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    return df


def feature_engineer(df):
    df = df.copy()

    # Derived features
    df['AvgMonthlyCharge'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['ChargePerService'] = df['MonthlyCharges'] / (
        (df['PhoneService'] == 'Yes').astype(int) +
        (df['InternetService'] != 'No').astype(int) +
        (df['OnlineSecurity'] == 'Yes').astype(int) +
        (df['OnlineBackup'] == 'Yes').astype(int) +
        (df['DeviceProtection'] == 'Yes').astype(int) +
        (df['TechSupport'] == 'Yes').astype(int) +
        (df['StreamingTV'] == 'Yes').astype(int) +
        (df['StreamingMovies'] == 'Yes').astype(int) + 1
    )
    df['NumServices'] = (
        (df['PhoneService'] == 'Yes').astype(int) +
        (df['MultipleLines'] == 'Yes').astype(int) +
        (df['InternetService'] != 'No').astype(int) +
        (df['OnlineSecurity'] == 'Yes').astype(int) +
        (df['OnlineBackup'] == 'Yes').astype(int) +
        (df['DeviceProtection'] == 'Yes').astype(int) +
        (df['TechSupport'] == 'Yes').astype(int) +
        (df['StreamingTV'] == 'Yes').astype(int) +
        (df['StreamingMovies'] == 'Yes').astype(int)
    )
    df['TenureGroup'] = pd.cut(df['tenure'],
                                bins=[0, 12, 24, 48, 72],
                                labels=['0-1yr', '1-2yr', '2-4yr', '4+yr'])
    df['IsAutoPayment'] = df['PaymentMethod'].isin(
        ['Bank transfer (automatic)', 'Credit card (automatic)']).astype(int)
    df['HasSecurityServices'] = (
        (df['OnlineSecurity'] == 'Yes') | (df['TechSupport'] == 'Yes')
    ).astype(int)

    return df


def encode_and_scale(df):
    df = df.copy()

    # Binary encoding (Yes/No → 1/0)
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling']
    binary_map  = {'Male': 1, 'Female': 0, 'Yes': 1, 'No': 0}
    for col in binary_cols:
        df[col] = df[col].map(binary_map)

    # One-hot encode multi-category columns
    ohe_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract',
                'PaymentMethod', 'TenureGroup']
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    # Convert bool columns to int
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def preprocess_pipeline(path='data/telecom_churn.csv', save_dir='models'):
    print("=" * 55)
    print("PREPROCESSING PIPELINE")
    print("=" * 55)

    df = load_and_clean(path)
    print(f"Loaded: {df.shape}")

    df = feature_engineer(df)
    print(f"After feature engineering: {df.shape}")

    df = encode_and_scale(df)
    print(f"After encoding: {df.shape}")

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges',
                'AvgMonthlyCharge', 'ChargePerService', 'NumServices']
    num_cols = [c for c in num_cols if c in X_train.columns]

    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(scaler, f'{save_dir}/scaler.pkl')

    X_train.to_csv(f'{save_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{save_dir}/X_test.csv',  index=False)
    y_train.to_csv(f'{save_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{save_dir}/y_test.csv',  index=False)

    print(f"\nTrain set : {X_train.shape}  |  Churn rate: {y_train.mean()*100:.1f}%")
    print(f"Test  set : {X_test.shape}   |  Churn rate: {y_test.mean()*100:.1f}%")
    print(f"\nFeatures  : {list(X_train.columns)}")
    print(f"\n[Saved] Preprocessed data & scaler to '{save_dir}/'")

    return X_train, X_test, y_train, y_test, scaler


if __name__ == '__main__':
    preprocess_pipeline()
