"""
Telecom Customer Churn Prediction
Step 3: Model Training — Logistic Regression, Random Forest, XGBoost
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm           import SVC
from sklearn.neighbors     import KNeighborsClassifier
from xgboost               import XGBClassifier
from imblearn.over_sampling import SMOTE

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, ConfusionMatrixDisplay
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

sns.set_theme(style='whitegrid')
plt.rcParams.update({'figure.dpi': 120})


def load_data(data_dir='models'):
    X_train = pd.read_csv(f'{data_dir}/X_train.csv')
    X_test  = pd.read_csv(f'{data_dir}/X_test.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv').squeeze()
    y_test  = pd.read_csv(f'{data_dir}/y_test.csv').squeeze()
    return X_train, X_test, y_train, y_test


def get_models():
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, C=1.0, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            random_state=42, class_weight='balanced', n_jobs=-1),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=2, random_state=42,
            use_label_encoder=False, eval_metric='logloss'),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=9, metric='euclidean'),
    }


def evaluate(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    return {
        'Accuracy' : round(accuracy_score(y_test, y_pred) * 100, 2),
        'Precision': round(precision_score(y_test, y_pred) * 100, 2),
        'Recall'   : round(recall_score(y_test, y_pred) * 100, 2),
        'F1 Score' : round(f1_score(y_test, y_pred) * 100, 2),
        'ROC-AUC'  : round(roc_auc_score(y_test, y_prob) * 100, 2) if y_prob is not None else 'N/A',
    }


def train_all_models(X_train, X_test, y_train, y_test,
                     save_dir='models', fig_dir='reports/figures'):
    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    os.makedirs(fig_dir, exist_ok=True)

    # SMOTE to handle class imbalance
    sm = SMOTE(random_state=42)
    X_tr_res, y_tr_res = sm.fit_resample(X_train, y_train)
    print(f"\nAfter SMOTE — Train: {X_tr_res.shape}, Churn rate: {y_tr_res.mean()*100:.1f}%")

    models  = get_models()
    results = {}
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\nTraining: {name} ...")
        model.fit(X_tr_res, y_tr_res)
        metrics = evaluate(model, X_test, y_test)
        cv_f1   = cross_val_score(model, X_train, y_train, cv=cv,
                                  scoring='f1', n_jobs=-1)
        metrics['CV F1 Mean'] = round(cv_f1.mean() * 100, 2)
        metrics['CV F1 Std']  = round(cv_f1.std()  * 100, 2)
        results[name] = metrics
        joblib.dump(model, f'{save_dir}/{name.replace(" ", "_")}.pkl')
        print(f"  Accuracy={metrics['Accuracy']}%  F1={metrics['F1 Score']}%  AUC={metrics['ROC-AUC']}%")

    # ── Results Table ──────────────────────────────────────────────────────
    results_df = pd.DataFrame(results).T
    results_df.to_csv(f'{save_dir}/model_results.csv')
    print("\n\nCOMPARISON TABLE:")
    print(results_df.to_string())

    # ── Figure: Model Comparison ───────────────────────────────────────────
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    plot_df = results_df[metrics_to_plot].astype(float)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(plot_df.index))
    width = 0.15
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
        bars = ax.bar(x + i*width, plot_df[metric], width, label=metric, color=color, alpha=0.85)

    ax.set_xticks(x + width*2)
    ax.set_xticklabels(plot_df.index, rotation=15, ha='right')
    ax.set_ylabel('Score (%)')
    ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(50, 105)
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/06_model_comparison.png', bbox_inches='tight')
    plt.close()
    print("\n[Saved] 06_model_comparison.png")

    # ── Figure: ROC Curves ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc:.2f})')
    ax.plot([0,1], [0,1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — All Models', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/07_roc_curves.png', bbox_inches='tight')
    plt.close()
    print("[Saved] 07_roc_curves.png")

    # ── Figure: Confusion Matrix (Best Model = XGBoost) ───────────────────
    best_model = models['XGBoost']
    cm = confusion_matrix(y_test, best_model.predict(X_test))
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Churn', 'Churn'])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('XGBoost — Confusion Matrix', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/08_confusion_matrix.png', bbox_inches='tight')
    plt.close()
    print("[Saved] 08_confusion_matrix.png")

    # ── Figure: Feature Importance (XGBoost) ─────────────────────────────
    feat_imp = pd.Series(
        best_model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(8, 7))
    feat_imp[::-1].plot(kind='barh', ax=ax, color='#2980b9', alpha=0.85)
    ax.set_title('Top 20 Feature Importances (XGBoost)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/09_feature_importance.png', bbox_inches='tight')
    plt.close()
    print("[Saved] 09_feature_importance.png")

    return models, results_df


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    train_all_models(X_train, X_test, y_train, y_test)
