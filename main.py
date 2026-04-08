import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("/Users/anshulshukla/Desktop/Credit risk main/dataset/credit_risk_dataset.csv")

# Define numerical and categorical columns
num_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

# Handle missing values for numerical features
imputer = SimpleImputer(strategy="median")
df[num_cols] = imputer.fit_transform(df[num_cols])

# Encode categorical variables
encoder = OneHotEncoder(drop='first', sparse_output=False)  # Fixed sparse=False issue
encoded_cats = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))

# Scale numerical features
scaler = StandardScaler()
scaled_nums = scaler.fit_transform(df[num_cols])
scaled_df = pd.DataFrame(scaled_nums, columns=num_cols)

# Prepare final dataset
X = pd.concat([scaled_df, encoded_df], axis=1)
y = df['loan_status']

# Save preprocessed dataset
X.to_csv("credit_risk_processed.csv", index=False)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
log_reg_preds = log_reg.predict(X_test)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)  # FIXED: Added missing prediction line

# Save the model
joblib.dump(rf, "random_forest_model.pkl")

# Save y_test and predictions for visualization
pd.DataFrame(y_test, columns=['loan_status']).to_csv("y_test.csv", index=False)
pd.DataFrame(rf_preds, columns=['prediction']).to_csv("y_pred.csv", index=False)

# Evaluate models
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_reg_preds))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))
print("\nClassification Report (Random Forest):\n", classification_report(y_test, rf_preds))
