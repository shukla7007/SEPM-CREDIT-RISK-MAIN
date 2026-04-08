import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure the static directory exists
if not os.path.exists("static"):
    os.makedirs("static")

# Load dataset
df = pd.read_csv("/Users/anshulshukla/Desktop/Credit risk main/dataset/credit_risk_dataset.csv")

# Identify categorical and numerical columns
categorical_cols = ['loan_grade', 'cb_person_default_on_file', 'loan_intent', 'person_home_ownership']  # Add other categorical columns as needed
numerical_cols = [col for col in df.columns if col not in categorical_cols and col != 'loan_status']

# Preprocess data
df = df.dropna()  # Handle missing values
# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)  # drop_first to avoid multicollinearity

# Separate features and target
X = df_encoded.drop("loan_status", axis=1)
y = df_encoded["loan_status"]

# Train-test split to recreate test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale only numerical features (after encoding, all columns are numerical)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save test data for future use
pd.DataFrame(y_test, columns=["loan_status"]).to_csv("y_test.csv", index=False)
pd.DataFrame(X_test, columns=X.columns).to_csv("X_test.csv", index=False)

# Load the trained Random Forest model
rf_model = joblib.load("random_forest_model.pkl")

# Generate predictions
y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
pd.DataFrame(y_pred, columns=["prediction"]).to_csv("y_pred.csv", index=False)

# Loan Status Pie Chart (Interactive with Plotly)
loan_status_counts = df['loan_status'].value_counts()
fig1 = go.Figure(data=[
    go.Pie(labels=loan_status_counts.index, values=loan_status_counts.values,
           marker=dict(colors=['lightblue', 'salmon']), textinfo='percent+label')
])
fig1.update_layout(title="Loan Status Distribution")
fig1.write_html("static/loan_status_pie.html")
plt.figure(figsize=(6, 6))
df['loan_status'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['lightblue', 'salmon'])
plt.title("Loan Status Distribution")
plt.ylabel("")
plt.savefig("static/loan_status_pie.png")
plt.close()

# Feature Importance from Random Forest (Interactive with Plotly)
feature_importance = rf_model.feature_importances_
features = X.columns  # Use the column names from the encoded features
fig2 = px.bar(x=feature_importance, y=features, orientation='h',
              labels={'x': 'Importance Score', 'y': 'Features'},
              title="Feature Importance (Random Forest)",
              color=feature_importance, color_continuous_scale='Viridis')
fig2.update_layout(yaxis={'tickmode': 'linear'})
fig2.write_html("static/feature_importance_bar.html")
plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importance, y=features, hue=features, palette="viridis", legend=False)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.savefig("static/feature_importance_bar.png")
plt.close()

# Confusion Matrix (Interactive with Plotly)
conf_matrix = confusion_matrix(y_test, y_pred)
fig3 = go.Figure(data=go.Heatmap(
    z=conf_matrix,
    x=["No Default", "Default"],
    y=["No Default", "Default"],
    colorscale="Blues",
    text=conf_matrix,
    texttemplate="%{text}",
    hoverinfo="z"
))
fig3.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
fig3.write_html("static/confusion_matrix_heatmap.html")
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Default", "Default"], yticklabels=["No Default", "Default"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("static/confusion_matrix_heatmap.png")
plt.close()

# ROC Curve (Interactive with Plotly)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {roc_auc:.2f})', line=dict(color='blue', width=2)))
fig4.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(color='grey', dash='dash')))
fig4.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", showlegend=True)
fig4.write_html("static/roc_curve.html")
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("static/roc_curve.png")
plt.close()

# Display the interactive plots
fig1.show()
fig2.show()
fig3.show()
fig4.show()

# Debug: Print shapes
print("y_test shape:", y_test.shape)
print("y_pred shape:", y_pred.shape)