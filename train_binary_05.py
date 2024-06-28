import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report

# Pre-tuned binary ensemble models
binary_ensemble_models = [
    ("Random Forest", RandomForestClassifier(
        max_depth=30, min_samples_split=10, n_estimators=200)),
    ("XGBoost", XGBClassifier(
        learning_rate=0.1, max_depth=3, n_estimators=300)),
    ("LightGBM", LGBMClassifier(
        learning_rate=0.3, max_depth=20, num_leaves=62)),
    ("CatBoost", CatBoostClassifier(
        depth=6, iterations=300, learning_rate=0.3, logging_level="Silent")),
    ("AdaBoost", AdaBoostClassifier(learning_rate=1.0, n_estimators=100)),
]

RAW_DIR = "data/raw"


dataset = pd.read_csv(f"{RAW_DIR}/fetal_health.csv")
X = dataset.drop('fetal_health', axis=1)
y = dataset['fetal_health']


# Create binary labels (0 for Normal, 1 for others)
y_binary = np.where(y == 1, 0, 1)  # 0 is Normal, 1 is for classes 2 and 3

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42)
_, _, y_train_binary, y_test_binary = train_test_split(
    X, y_binary, test_size=0.5, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train binary ensemble
binary_ensemble = VotingClassifier(
    estimators=binary_ensemble_models, voting='hard')
binary_ensemble.fit(X_train_scaled, y_train_binary)

# Prepare data for joined classes
X_joined = X[y != 1]  # Select non-Normal instances
y_joined = y[y != 1]  # Select labels for non-Normal instances

# Use LabelEncoder to transform labels to 0 and 1
le = LabelEncoder()
y_joined_encoded = le.fit_transform(y_joined)

X_train_joined, X_test_joined, y_train_joined, y_test_joined = train_test_split(
    X_joined, y_joined_encoded, test_size=0.5, random_state=42)

X_train_joined_scaled = scaler.transform(X_train_joined)
X_test_joined_scaled = scaler.transform(X_test_joined)

# Create joined ensemble with best parameters
joined_ensemble_models = [
    ("AdaBoost", AdaBoostClassifier(learning_rate=0.1, n_estimators=50)),
    ("Random Forest", RandomForestClassifier(
        max_depth=30, min_samples_split=5, n_estimators=100)),
    ("XGBoost", XGBClassifier(learning_rate=0.1, max_depth=6, n_estimators=100)),
    ("LightGBM", LGBMClassifier(learning_rate=0.1, max_depth=20, num_leaves=31)),
    ("CatBoost", CatBoostClassifier(depth=4, iterations=300,
     learning_rate=0.01, logging_level="Silent")),
]

# Create and train joined ensemble
joined_ensemble = VotingClassifier(
    estimators=joined_ensemble_models, voting='hard')
joined_ensemble.fit(X_train_joined_scaled, y_train_joined)

# Evaluate ensembles


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, y_pred))


evaluate_model(binary_ensemble, X_test_scaled,
               y_test_binary, "Binary Ensemble")
evaluate_model(joined_ensemble, X_test_joined_scaled,
               y_test_joined, "Joined Ensemble")

# Function to combine predictions using ensembles


def predict_combined_ensemble(X_new, binary_ensemble, joined_ensemble, le):
    X_new_scaled = scaler.transform(X_new)
    binary_pred = binary_ensemble.predict(X_new_scaled)
    joined_mask = binary_pred == 1
    # Initialize with 1 for Normal, 0 for others
    final_pred = np.where(binary_pred == 0, 1, 0)
    if np.any(joined_mask):
        joined_pred = joined_ensemble.predict(X_new_scaled[joined_mask])
        final_pred[joined_mask] = le.inverse_transform(joined_pred)
    return final_pred


# Evaluate combined ensemble
y_pred_combined = predict_combined_ensemble(
    X_test, binary_ensemble, joined_ensemble, le)
print("\nClassification Report for Combined Ensemble:")
print(classification_report(y_test, y_pred_combined))
