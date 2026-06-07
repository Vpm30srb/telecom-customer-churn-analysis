import pandas as pd
import numpy as np

np.random.seed(42)
n = 7043

# Customer demographics
customer_id = ['CUST' + str(i).zfill(5) for i in range(1, n+1)]
gender = np.random.choice(['Male', 'Female'], n)
senior_citizen = np.random.choice([0, 1], n, p=[0.84, 0.16])
partner = np.random.choice(['Yes', 'No'], n, p=[0.48, 0.52])
dependents = np.random.choice(['Yes', 'No'], n, p=[0.30, 0.70])

# Tenure (months)
tenure = np.random.randint(1, 73, n)

# Services
phone_service = np.random.choice(['Yes', 'No'], n, p=[0.90, 0.10])
multiple_lines = np.where(phone_service == 'No', 'No phone service',
                 np.random.choice(['Yes', 'No'], n, p=[0.42, 0.58]))
internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.34, 0.44, 0.22])

def internet_add_on(internet_service, yes_p=0.44):
    return np.where(internet_service == 'No', 'No internet service',
            np.random.choice(['Yes', 'No'], n, p=[yes_p, 1-yes_p]))

online_security   = internet_add_on(internet_service, 0.29)
online_backup     = internet_add_on(internet_service, 0.34)
device_protection = internet_add_on(internet_service, 0.34)
tech_support      = internet_add_on(internet_service, 0.29)
streaming_tv      = internet_add_on(internet_service, 0.38)
streaming_movies  = internet_add_on(internet_service, 0.39)

# Contract & billing
contract         = np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.21, 0.24])
paperless_billing= np.random.choice(['Yes', 'No'], n, p=[0.59, 0.41])
payment_method   = np.random.choice(
    ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
    n, p=[0.34, 0.23, 0.22, 0.21])

# Charges
monthly_charges = np.round(
    20 + (internet_service == 'Fiber optic') * 30
      + (internet_service == 'DSL') * 15
      + (multiple_lines == 'Yes') * 10
      + (online_security == 'Yes') * 5
      + (online_backup == 'Yes') * 5
      + (device_protection == 'Yes') * 5
      + (tech_support == 'Yes') * 5
      + (streaming_tv == 'Yes') * 8
      + (streaming_movies == 'Yes') * 8
      + np.random.normal(0, 5, n), 2)
monthly_charges = np.clip(monthly_charges, 18.25, 118.75)

total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 10, n), 2)
total_charges = np.clip(total_charges, 18.8, 8684.8)

# Churn logic — realistic probability model
churn_prob = (
    0.05
    + 0.20 * (contract == 'Month-to-month')
    + 0.25 * (internet_service == 'Fiber optic')
    - 0.10 * (internet_service == 'No')
    + 0.08 * (payment_method == 'Electronic check')
    - 0.15 * (tenure > 36)
    - 0.10 * (tenure > 60)
    + 0.05 * (senior_citizen == 1)
    - 0.05 * (partner == 'Yes')
    - 0.05 * (dependents == 'Yes')
    - 0.05 * (online_security == 'Yes')
    - 0.04 * (tech_support == 'Yes')
    + 0.03 * (paperless_billing == 'Yes')
    + np.random.normal(0, 0.05, n)
)
churn_prob = np.clip(churn_prob, 0.02, 0.95)
churn = np.where(np.random.rand(n) < churn_prob, 'Yes', 'No')

df = pd.DataFrame({
    'customerID': customer_id,
    'gender': gender,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Churn': churn
})

df.to_csv('/home/claude/telecom_churn/data/telecom_churn.csv', index=False)
print(f"Dataset saved: {df.shape}")
print(df['Churn'].value_counts())
print(df.head(3))
