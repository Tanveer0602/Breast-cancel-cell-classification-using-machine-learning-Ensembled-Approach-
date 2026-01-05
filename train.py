# train.py
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 1) Load dataset
data = load_breast_cancer()
X_full = pd.DataFrame(data.data, columns=data.feature_names)  # 30 features
y = data.target
target_names = data.target_names  # ['malignant', 'benign']

print("Original shape:", X_full.shape)

# 2) Select top-10 features (and remember their names)
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_full, y)   # shape -> (569, 10)
mask = selector.get_support()                    # boolean mask length 30
selected_feature_names = X_full.columns[mask].tolist()
print("Selected features (10):", selected_feature_names)

# 3) Train/test split on the reduced matrix
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Fit scaler (on the 10 features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5) Train 3 models on the 10-feature scaled data
lr = LogisticRegression(max_iter=500, random_state=42)
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=150, random_state=42)

lr.fit(X_train_scaled, y_train)
dt.fit(X_train_scaled, y_train)
rf.fit(X_train_scaled, y_train)

# 6) Quick evaluation (on test set)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print("LR acc:", accuracy_score(y_test, lr.predict(X_test_scaled)))
print("DT acc:", accuracy_score(y_test, dt.predict(X_test_scaled)))
print("RF acc:", accuracy_score(y_test, rf.predict(X_test_scaled)))

print("\nLogistic Regression report:\n", classification_report(y_test, lr.predict(X_test_scaled)))
print("Decision Tree report:\n", classification_report(y_test, dt.predict(X_test_scaled)))
print("Random Forest report:\n", classification_report(y_test, rf.predict(X_test_scaled)))

# 7) Save everything required by the Flask app:
#    - list of selected feature names (so frontend labels can be meaningful)
#    - scaler (fitted on 10 features)
#    - the three models
#    - target mapping (so we know which label = malignant/benign)
artifact = {
    "feature_names": selected_feature_names,
    "scaler": scaler,
    "models": {
        "lr": lr,
        "dt": dt,
        "rf": rf
    },
    "target_names": target_names  # index 0 -> 'malignant', index 1 -> 'benign'
}

with open("artifact_10feat.pkl", "wb") as f:
    pickle.dump(artifact, f)

print("\nSaved artifact_10feat.pkl (features, scaler, models, target_names).")
print("Training complete.")
