# %%
# random forest classifier
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.ensemble import VotingClassifier
import warnings
from sklearn.preprocessing import StandardScaler
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# classification report

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt


# f1, precision, recall, weighted accuracy
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    make_scorer,
    balanced_accuracy_score,
)

# xgboost classifier

from xgboost import XGBClassifier

# lightgbm classifier

from lightgbm import LGBMClassifier

# catboost classifier

from catboost import CatBoostClassifier

import pandas as pd

# Adaboot classifier
from sklearn.ensemble import AdaBoostClassifier

# SVM

from sklearn.svm import SVC

# KNN

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

# %%
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

# %%
dataset = pd.read_csv(f"{RAW_DIR}/fetal_health.csv")

# %% [markdown]
# ## Removing histogram features
# 'histogram_width',
#        'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
#        'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean',
#        'histogram_median', 'histogram_variance', 'histogram_tendency',

# %%
dataset.columns

# %%
label2class = {
    1: "Normal",
    2: "Suspect",
    3: "Pathology"
}

labeel2class_encoded = {
    0: "Normal",
    1: "Suspect",
    2: "Pathology"
}

# %%
# split (train, test) stratified and save into processed directory

train, test = train_test_split(
    dataset, test_size=0.5, stratify=dataset["fetal_health"], random_state=42
)

train.to_csv(f"{PROCESSED_DIR}/train.csv", index=False)

test.to_csv(f"{PROCESSED_DIR}/test.csv", index=False)

# %%
x_train = train.drop("fetal_health", axis=1)
y_train = train["fetal_health"]

y_train = LabelEncoder().fit_transform(y_train)

# %%
x_test = test.drop("fetal_health", axis=1)

y_test = test["fetal_health"]

y_test = LabelEncoder().fit_transform(y_test)

# %%
x_train_flipped = x_test.copy()
x_test_flipped = x_train.copy()

y_train_flipped = y_test.copy()
y_test_flipped = y_train.copy()

# %%

scaler = StandardScaler()

# %%
# Dropping 'severe_decelerations' from the training and testing datasets
x_train_no_severe = x_train.drop('severe_decelerations', axis=1)
x_test_no_severe = x_test.drop('severe_decelerations', axis=1)

x_train_no_severe_scaled = scaler.fit_transform(x_train_no_severe)
x_test_no_severe_scaled = scaler.transform(x_test_no_severe)

# %%
# Copying the datasets to keep the 'severe_decelerations' column separate
x_train_severe = x_train.copy()
x_test_severe = x_test.copy()

# Removing 'severe_decelerations' temporarily for scaling
severe_train = x_train_severe.pop('severe_decelerations')
severe_test = x_test_severe.pop('severe_decelerations')

# Scaling the remaining features
scaler_severe = StandardScaler()
x_train_severe_scaled = scaler_severe.fit_transform(x_train_severe)
x_test_severe_scaled = scaler_severe.transform(x_test_severe)

# Adding the unscaled 'severe_decelerations' back to the scaled datasets

x_train_severe_scaled = np.column_stack((x_train_severe_scaled, severe_train))

x_test_severe_scaled = np.column_stack((x_test_severe_scaled, severe_test))

# %%
x_train_scaled = scaler.fit_transform(x_train)

x_test_scaled = scaler.transform(x_test)

# %%
models = [
    ("Random Forest", RandomForestClassifier()),
    ("XGBoost", XGBClassifier()),
    ("LightGBM", LGBMClassifier(verbosity=-1)),
    ("CatBoost", CatBoostClassifier(logging_level="Silent")),
    ("AdaBoost", AdaBoostClassifier()),
    ("SVM", SVC()),
    ("KNN", KNeighborsClassifier()),
]

# %%
FOLDS = 10

# %%
# Define the models and their corresponding parameter grids
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
    "SVM": {
        "model": SVC(),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
            "decision_function_shape": ["ovo", "ovr"]
        }
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree"],
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
}

# %% [markdown]
# ### With Histogram

# %%
# remove warnings

warnings.filterwarnings("ignore")

# %%
# Apply GridSearchCV for each model


def train_models(y_train, x_train_scaled, FOLDS, models_params):
    best_estimators = {}
    for model_name, mp in models_params.items():
        print(f"Running GridSearchCV for {model_name}")
        grid_search = GridSearchCV(
            mp['model'], mp['params'], cv=FOLDS, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(x_train_scaled,
                        y_train)
        best_estimators[model_name] = grid_search.best_estimator_
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best score for {model_name}: {grid_search.best_score_}")
    return best_estimators


best_estimators_regular = train_models(
    y_train, x_train_scaled, FOLDS, models_params)

best_estimators_no_severe = train_models(
    y_train, x_train_no_severe_scaled, FOLDS, models_params)

best_estimators_severe = train_models(
    y_train, x_train_severe_scaled, FOLDS, models_params)


best_estimators_regular_flipped = train_models(
    y_train_flipped, x_train_scaled, FOLDS, models_params)


# # %%
# print(best_estimators)

# # %% [markdown]
# # ## Tests

# # %%

# # Create a list of tuples (classifier name, classifier object)
# classifiers = [
#     ("Random Forest", best_estimators["Random Forest"]),
#     ("XGBoost", best_estimators["XGBoost"]),
#     ("LightGBM", best_estimators["LightGBM"]),

# ]

# # %%
# trained_classifiers = {}
# for clf_name, clf in classifiers:
#     clf.fit(x_train, y_train)
#     trained_classifiers[clf_name] = clf

# # Create a voting classifier
# voting_clf = VotingClassifier(estimators=classifiers, voting='soft')

# # Fit the voting classifier

# voting_clf.fit(x_train, y_train)

# # Predict the test set

# y_pred = voting_clf.predict(x_test)

# # %%
# print(classification_report(y_test, y_pred))

# # %%
# # matrix confusion

# c = confusion_matrix(y_test, y_pred)

# # plot

# plt.figure(figsize=(10, 5))

# sns.heatmap(c, annot=True, fmt="d", cmap="Blues",
#             xticklabels=[label2class[i] for i in range(1, 4)],
#             yticklabels=[label2class[i] for i in range(1, 4)])

# plt.title(f"Confusion Matrix for Ensemble Model")
# plt.xlabel("Predicted")
# plt.ylabel("True")


# plt.show()

# # %%
# # smote

# # %%

# smote = ADASYN(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

# # %%
# trained_classifiers_unb = {}
# for clf_name, clf in classifiers:
#     clf.fit(X_train_resampled, y_train_resampled)
#     trained_classifiers[clf_name] = clf

# # Create a voting classifier
# voting_clf_unb = VotingClassifier(estimators=classifiers, voting='soft')

# # Fit the voting classifier

# voting_clf_unb.fit(X_train_resampled, y_train_resampled)

# # Predict the test set

# y_pred_unb = voting_clf.predict(x_test_scaled)

# # %%
# print(classification_report(y_test, y_pred_unb))

# # %%
# print(classification_report(y_test, y_pred))

# # %%

# # %%
# # Apply BorderlineSMOTE

# smote = BorderlineSMOTE(random_state=42)

# X_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

# trained_classifiers_unb = {}
# for clf_name, clf in classifiers:
#     clf.fit(X_train_resampled, y_train_resampled)
#     trained_classifiers[clf_name] = clf

# # Create a voting classifier
# voting_clf_unb = VotingClassifier(estimators=classifiers, voting='soft')

# # Fit the voting classifier

# voting_clf_unb.fit(X_train_resampled, y_train_resampled)

# # Predict the test set

# y_pred_unb = voting_clf.predict(x_test)


# print(classification_report(y_test, y_pred_unb))

# # %%
# print(classification_report(y_test, y_pred_unb))

# # %%
# print(classification_report(y_test, y_pred))
