import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load dataset
df = pd.read_csv("/Users/anshulshukla/Desktop/Credit risk main/dataset/credit_risk_dataset.csv")

# Print actual column names
print("Dataset Columns:", df.columns)

# Strip spaces from column names if necessary
df.columns = df.columns.str.strip()

# Define categorical and numerical columns
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
num_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']

# Check if categorical columns exist in the dataset
missing_cols = [col for col in categorical_cols if col not in df.columns]
if missing_cols:
    print(f"⚠️ Missing Columns: {missing_cols}")
else:
    # Proceed only if all columns exist
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoder.fit(df[categorical_cols])
    
    scaler = StandardScaler()
    scaler.fit(df[num_cols])
    
    print("✅ Encoder and Scaler trained successfully.")
# Save the preprocessor objects
joblib.dump(encoder, "encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
