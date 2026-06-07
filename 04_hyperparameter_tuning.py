"""
Telecom Customer Churn Prediction
Step 4: Hyperparameter Tuning (Best Model — XGBoost)
"""

import pandas as pd
import numpy as np
import joblib, warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, classification_report


def load_data(data_dir='models'):
    X_train = pd.read_csv(f'{data_dir}/X_train.csv')
    X_test  = pd.read_csv(f'{data_dir}/X_test.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv').squeeze()
    y_test  = pd.read_csv(f'{data_dir}/y_test.csv').squeeze()
    return X_train, X_test, y_train, y_test


def tune_xgboost(X_train, y_train, save_dir='models'):
    print("=" * 55)
    print("HYPERPARAMETER TUNING — XGBoost")
    print("=" * 55)

    param_grid = {
        'n_estimators'    : [100, 200, 300],
        'max_depth'       : [4, 6, 8],
        'learning_rate'   : [0.03, 0.05, 0.1],
        'subsample'       : [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8],
        'scale_pos_weight': [2, 3],
    }

    base = XGBClassifier(
        random_state=42, use_label_encoder=False,
        eval_metric='logloss', n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        base, param_grid,
        scoring='f1', cv=cv, n_jobs=-1, verbose=1
    )

    print("\nRunning GridSearchCV (this may take a few minutes)...")
    grid_search.fit(X_train, y_train)

    print(f"\nBest Params : {grid_search.best_params_}")
    print(f"Best CV F1  : {grid_search.best_score_*100:.2f}%")

    joblib.dump(grid_search.best_estimator_, f'{save_dir}/XGBoost_tuned.pkl')
    print(f"\n[Saved] Tuned model → {save_dir}/XGBoost_tuned.pkl")

    return grid_search.best_estimator_


def evaluate_tuned(model, X_test, y_test):
    print("\n" + "=" * 55)
    print("TUNED MODEL — TEST SET EVALUATION")
    print("=" * 55)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\nF1 Score  : {f1_score(y_test, y_pred)*100:.2f}%")
    print(f"ROC-AUC   : {roc_auc_score(y_test, y_prob)*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    best_model = tune_xgboost(X_train, y_train)
    evaluate_tuned(best_model, X_test, y_test)
