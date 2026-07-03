# Logistic Regression - Diabetes Prediction

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt

# Step 1: Load dataset
# Replace with your actual file path
df = pd.read_csv("diabetes.csv")

# Step 2: Assign column names (if not already present)
df.columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

# Step 3: Check for missing or zero values
print("Missing values:\n", df.isnull().sum())
print("Zero values per column:\n", (df == 0).sum())

# Step 4: Split dataset
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5: Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Train Logistic Regression model
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)

# Step 7: Predictions and evaluation
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("🎯 Precision:", precision_score(y_test, y_pred))
print("🔎 Recall:", recall_score(y_test, y_pred))
print("⚖️ F1-score:", f1_score(y_test, y_pred))
print("📈 ROC-AUC:", roc_auc_score(y_test, y_pred_proba))

# Step 8: Interpret coefficients
coefficients = pd.DataFrame({
    "Feature": df.columns[:-1],
    "Coefficient": logreg.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print("\n📌 Model Coefficients:\n", coefficients)

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_pred_proba):.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.show()
