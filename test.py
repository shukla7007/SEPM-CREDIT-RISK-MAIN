import pandas as pd
import joblib

y_test = pd.read_csv("y_test.csv")
X_test = pd.read_csv("X_test.csv")
rf_model = joblib.load("random_forest_model.pkl")

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)