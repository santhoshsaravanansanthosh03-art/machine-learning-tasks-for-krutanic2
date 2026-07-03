# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_score, recall_score, f1_score
)
from xgboost import XGBClassifier

# Step 2: Load dataset
# Assuming you have titanic.csv with columns like: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
df = pd.read_csv("titanic.csv")

# Step 3: Assign appropriate column names (if needed)
df.columns = ["PassengerId","Survived","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]

# Step 4: Check for missing or zero values
print(df.isnull().sum())
print((df == 0).sum())

# Handle missing values (simple strategy: fill Age with median, Embarked with mode)
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Drop columns not useful for prediction (Name, Ticket, Cabin, PassengerId)
df = df.drop(["Name","Ticket","Cabin","PassengerId"], axis=1)

# Encode categorical variables
df["Sex"] = df["Sex"].map({"male":0,"female":1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# Step 5: Split dataset
X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 6: Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Train XGBoost model with parameters
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)
xgb.fit(X_train, y_train)

# Step 8: Predictions
y_pred = xgb.predict(X_test)
y_pred_proba = xgb.predict_proba(X_test)[:,1]

# Step 9: Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("✅ Accuracy:", accuracy)
print("\n📊 Confusion Matrix:\n", conf_matrix)
print("\n📈 Classification Report:\n", classification_report(y_test, y_pred))
print("🎯 Precision:", precision)
print("🔎 Recall:", recall)
print("⚖️ F1-score:", f1)
print("🔹 ROC-AUC Score:", roc_auc)

# Step 10: Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f"XGBoost (AUC = {roc_auc:.2f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Step 11: Interpret model coefficients (feature importance)
importances = xgb.feature_importances_
features = X.columns
importance_df = pd.DataFrame({"Feature":features,"Importance":importances}).sort_values(by="Importance",ascending=False)
print("\n🔍 Feature Importances:\n", importance_df)
