# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt

# Step 2: Load dataset
df = pd.read_csv("pima_diabetes.csv")

# Step 3: Assign column names
df.columns = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness",
    "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"
]

# Step 4: Handle missing/unrealistic zero values
cols_with_zeros = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
df.fillna(df.median(), inplace=True)

# Step 5: Split dataset
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 6: Train Decision Tree
dt = DecisionTreeClassifier(
    criterion="gini", max_depth=4, random_state=42
)
dt.fit(X_train, y_train)

# Step 7: Predictions
y_pred = dt.predict(X_test)
y_pred_proba = dt.predict_proba(X_test)[:,1]

# Step 8: Evaluation
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📈 Classification Report:\n", classification_report(y_test, y_pred))
print("🎯 Precision:", precision_score(y_test, y_pred))
print("🔎 Recall:", recall_score(y_test, y_pred))
print("⚖️ F1-score:", f1_score(y_test, y_pred))
print("🔹 ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))

# Step 9: Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f"Decision Tree (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Step 10: Feature importance
importances = dt.feature_importances_
features = X.columns
importance_df = pd.DataFrame({"Feature":features,"Importance":importances}).sort_values(by="Importance",ascending=False)
print("\n🔍 Feature Importances:\n", importance_df)

# Optional: Visualize the tree
plt.figure(figsize=(12,8))
plot_tree(dt, feature_names=features, class_names=["No Diabetes","Diabetes"], filled=True)
plt.show()
