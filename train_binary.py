import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Function to train models for joined classes


def train_models(y_train, x_train_scaled, FOLDS, models_params):
    best_estimators = {}
    for model_name, mp in models_params.items():
        print(f"Running GridSearchCV for {model_name}")
        grid_search = GridSearchCV(
            mp['model'], mp['params'], cv=FOLDS, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(x_train_scaled, y_train)
        best_estimators[model_name] = grid_search.best_estimator_
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best score for {model_name}: {grid_search.best_score_}")
    return best_estimators


RAW_DIR = "data/raw"


dataset = pd.read_csv(f"{RAW_DIR}/fetal_health.csv")
X = dataset.drop('fetal_health', axis=1)
y = dataset['fetal_health']


# Create binary labels (0 for Normal, 1 for others)
y_binary = np.where(y == 1, 0, 1)  # 0 is Normal, 1 is for classes 2 and 3

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
_, _, y_train_binary, y_test_binary = train_test_split(
    X, y_binary, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train binary ensemble
binary_ensemble = VotingClassifier(
    estimators=binary_ensemble_models, voting='soft')
binary_ensemble.fit(X_train_scaled, y_train_binary)

# Prepare data for joined classes
X_joined = X[y != 1]  # Select non-Normal instances
y_joined = y[y != 1]  # Select labels for non-Normal instances

# Use LabelEncoder to transform labels to 0 and 1
le = LabelEncoder()
y_joined_encoded = le.fit_transform(y_joined)

X_train_joined, X_test_joined, y_train_joined, y_test_joined = train_test_split(
    X_joined, y_joined_encoded, test_size=0.2, random_state=42)

X_train_joined_scaled = scaler.transform(X_train_joined)
X_test_joined_scaled = scaler.transform(X_test_joined)

# Define models and parameters for joined classes
models_params = {
    "Random Forest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, 30],
            "min_samples_split": [2, 5, 10]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 0.3]
        }
    },
    "LightGBM": {
        "model": LGBMClassifier(verbosity=-1),
        "params": {
            "num_leaves": [31, 62, 128],
            "max_depth": [10, 20, 30],
            "learning_rate": [0.01, 0.1, 0.3]
        }
    },
    "CatBoost": {
        "model": CatBoostClassifier(logging_level="Silent"),
        "params": {
            "iterations": [100, 200, 300],
            "depth": [4, 6, 10],
            "learning_rate": [0.01, 0.1, 0.3]
        }
    },
    "AdaBoost": {
        "model": AdaBoostClassifier(),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 1.0],
        }
    },
}

FOLDS = 5
joined_classifiers = train_models(
    y_train_joined, X_train_joined_scaled, FOLDS, models_params)

# Create ensemble for joined classification
joined_ensemble = VotingClassifier(estimators=[(
    name, model) for name, model in joined_classifiers.items()], voting='soft')
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

# Classification Report for Joined Ensemble:
#               precision    recall  f1-score   support

#            0       0.98      1.00      0.99        53
#            1       1.00      0.98      0.99        42

#     accuracy                           0.99        95
#    macro avg       0.99      0.99      0.99        95
# weighted avg       0.99      0.99      0.99        95


# Classification Report for Combined Ensemble:
#               precision    recall  f1-score   support

#          1.0       0.97      0.98      0.98       333
#          2.0       0.92      0.88      0.90        64
#          3.0       1.00      0.97      0.98        29

#     accuracy                           0.97       426
#    macro avg       0.96      0.94      0.95       426
# weighted avg       0.97      0.97      0.97       426
