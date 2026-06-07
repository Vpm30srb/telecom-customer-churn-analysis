"""
Telecom Customer Churn Prediction
main.py — Run the full pipeline end-to-end
"""

import os
import sys

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.02_preprocessing import preprocess_pipeline
from src.03_train_models  import load_data, train_all_models

def run_pipeline():
    print("\n" + "="*60)
    print("  TELECOM CHURN PREDICTION — FULL PIPELINE")
    print("="*60)

    os.makedirs('models', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)

    print("\n[1/3] Preprocessing & Feature Engineering...")
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(
        path='data/telecom_churn.csv', save_dir='models'
    )

    print("\n[2/3] Training Models...")
    X_train, X_test, y_train, y_test = load_data('models')
    models, results = train_all_models(
        X_train, X_test, y_train, y_test,
        save_dir='models', fig_dir='reports/figures'
    )

    print("\n[3/3] Pipeline complete!")
    print("\nBest model (XGBoost) saved to: models/XGBoost.pkl")
    print("All evaluation plots saved to: reports/figures/")
    print("\nTo predict on new data, run:")
    print("  python src/05_predict.py\n")

if __name__ == '__main__':
    run_pipeline()
