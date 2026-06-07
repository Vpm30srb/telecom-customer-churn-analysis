"""
Telecom Customer Churn Prediction
Step 1: Exploratory Data Analysis & Preprocessing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'figure.dpi': 120, 'font.size': 11})

# ── Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv('data/telecom_churn.csv')
print("=" * 60)
print("TELECOM CHURN — EXPLORATORY DATA ANALYSIS")
print("=" * 60)
print(f"\nDataset Shape : {df.shape}")
print(f"Churn Rate    : {(df['Churn']=='Yes').mean()*100:.1f}%\n")
print(df.dtypes)
print("\nMissing Values:\n", df.isnull().sum()[df.isnull().sum() > 0])
print("\nBasic Stats:\n", df.describe())

# ── Figure 1: Churn Distribution ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Churn Distribution', fontsize=14, fontweight='bold')

counts = df['Churn'].value_counts()
colors = ['#2ecc71', '#e74c3c']
axes[0].pie(counts, labels=counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 12})
axes[0].set_title('Overall Churn Split')

sns.countplot(data=df, x='Churn', palette={'Yes': '#e74c3c', 'No': '#2ecc71'}, ax=axes[1])
axes[1].set_title('Churn Count')
axes[1].set_xlabel('Churn')
axes[1].set_ylabel('Count')
for p in axes[1].patches:
    axes[1].annotate(f'{int(p.get_height()):,}',
                     (p.get_x() + p.get_width()/2., p.get_height()),
                     ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.savefig('reports/figures/01_churn_distribution.png', bbox_inches='tight')
plt.close()
print("\n[Saved] 01_churn_distribution.png")

# ── Figure 2: Numerical Features vs Churn ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Numerical Features vs Churn', fontsize=14, fontweight='bold')

for ax, col in zip(axes, ['tenure', 'MonthlyCharges', 'TotalCharges']):
    sns.histplot(data=df, x=col, hue='Churn', kde=True,
                 palette={'Yes': '#e74c3c', 'No': '#3498db'}, ax=ax, alpha=0.6)
    ax.set_title(col)

plt.tight_layout()
plt.savefig('reports/figures/02_numerical_features.png', bbox_inches='tight')
plt.close()
print("[Saved] 02_numerical_features.png")

# ── Figure 3: Categorical Features vs Churn ───────────────────────────────
cat_cols = ['Contract', 'InternetService', 'PaymentMethod',
            'gender', 'SeniorCitizen', 'Partner']
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Categorical Features vs Churn Rate', fontsize=14, fontweight='bold')

for ax, col in zip(axes.flatten(), cat_cols):
    churn_rate = df.groupby(col)['Churn'].apply(lambda x: (x=='Yes').mean()*100).reset_index()
    churn_rate.columns = [col, 'ChurnRate']
    churn_rate = churn_rate.sort_values('ChurnRate', ascending=False)
    bars = ax.bar(churn_rate[col].astype(str), churn_rate['ChurnRate'],
                  color=plt.cm.RdYlGn_r(churn_rate['ChurnRate']/100))
    ax.set_title(col, fontsize=11, fontweight='bold')
    ax.set_ylabel('Churn Rate (%)')
    ax.tick_params(axis='x', rotation=15)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('reports/figures/03_categorical_churn_rates.png', bbox_inches='tight')
plt.close()
print("[Saved] 03_categorical_churn_rates.png")

# ── Figure 4: Correlation Heatmap ─────────────────────────────────────────
df_enc = df.copy()
df_enc['Churn_bin'] = (df_enc['Churn'] == 'Yes').astype(int)
for c in df_enc.select_dtypes(include='object').columns:
    df_enc[c] = pd.Categorical(df_enc[c]).codes

num_cols = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges','Churn_bin']
corr = df_enc[num_cols].corr()

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, ax=ax, linewidths=0.5)
ax.set_title('Correlation Matrix (Numeric Features)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/04_correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("[Saved] 04_correlation_heatmap.png")

# ── Figure 5: Tenure vs Monthly Charges scatter ───────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for label, color in [('Yes', '#e74c3c'), ('No', '#3498db')]:
    subset = df[df['Churn'] == label]
    ax.scatter(subset['tenure'], subset['MonthlyCharges'],
               alpha=0.3, s=15, c=color, label=f'Churn={label}')
ax.set_xlabel('Tenure (months)')
ax.set_ylabel('Monthly Charges ($)')
ax.set_title('Tenure vs Monthly Charges by Churn', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('reports/figures/05_tenure_vs_charges.png', bbox_inches='tight')
plt.close()
print("[Saved] 05_tenure_vs_charges.png")

print("\n[EDA Complete] All figures saved to reports/figures/")
