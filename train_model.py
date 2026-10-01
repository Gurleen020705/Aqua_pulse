import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


# ============================================================
# 1. LOAD GEMSTAT DATASET
# ============================================================

df = pd.read_csv("GEMStat_RF_Final.csv")

print("\n========================================")
print("       GEMSTAT DATASET")
print("========================================")

print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 2. CHECK DATA
# ============================================================

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nLabel distribution:")
print(df["Label"].value_counts())

print("\nLabel meaning:")
print("1 = Healthy")
print("0 = At-Risk")


# ============================================================
# 3. SELECT INPUT FEATURES
# ============================================================

features = [
    "DO",
    "Temperature",
    "pH",
    "Nitrate",
    "Phosphorus"
]

X = df[features]

# Target variable
y = df["Label"]

print("\nFeatures used:")
print(features)

print("\nTarget variable: Label")


# ============================================================
# 4. SPLIT DATA INTO TRAINING AND TESTING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========================================")
print("       TRAIN / TEST SPLIT")
print("========================================")

print("Total samples:", len(df))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# ============================================================
# 5. CREATE RANDOM FOREST CLASSIFIER
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

print("\n========================================")
print("       RANDOM FOREST TRAINING")
print("========================================")


# ============================================================
# 6. TRAIN MODEL
# ============================================================

rf_model.fit(X_train, y_train)

print("Random Forest training completed!")


# ============================================================
# 7. TEST MODEL ON 20% TESTING DATA
# ============================================================

y_pred = rf_model.predict(X_test)

print("\nModel predictions generated for test data.")


# ============================================================
# 8. CREATE TEST PREDICTION TABLE
# ============================================================

results = df.loc[X_test.index, [
    "GEMS Station Number",
    "Sample Date",
    *features
]].copy()

train_stations = set(df.loc[X_train.index, "GEMS Station Number"])
results["Station_Also_In_Train"] = results["GEMS Station Number"].isin(
    train_stations
)

# Actual labels
results["Actual_Label"] = y_test

# Model predictions
results["Predicted_Label"] = y_pred


# Convert labels into readable names
results["Actual_Status"] = results["Actual_Label"].map({
    1: "Healthy",
    0: "At-Risk"
})

results["Predicted_Status"] = results["Predicted_Label"].map({
    1: "Healthy",
    0: "At-Risk"
})

class_index = {int(label): i for i, label in enumerate(rf_model.classes_)}
probabilities = rf_model.predict_proba(X_test)
results["At_Risk_Probability"] = [
    round(float(row[class_index[0]]), 4) for row in probabilities
]
results["Healthy_Probability"] = [
    round(float(row[class_index[1]]), 4) for row in probabilities
]


# Check whether prediction is correct
results["Correct"] = (
    results["Actual_Label"] ==
    results["Predicted_Label"]
)


print("\n========================================")
print("       TEST DATA PREDICTIONS")
print("========================================")

print(results.to_string())


# ============================================================
# 9. COUNT CORRECT AND INCORRECT PREDICTIONS
# ============================================================

correct = results["Correct"].sum()
incorrect = (~results["Correct"]).sum()

print("\nCorrect predictions:", correct)
print("Incorrect predictions:", incorrect)

print("Total test samples:", len(results))


# ============================================================
# 10. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    pos_label=1
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label=1
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label=1
)


print("\n========================================")
print("       MODEL PERFORMANCE")
print("========================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("       CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "At-Risk",
            "Healthy"
        ]
    )
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n========================================")
print("       CONFUSION MATRIX")
print("========================================")

print(cm)


plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "At-Risk",
        "Healthy"
    ],
    yticklabels=[
        "At-Risk",
        "Healthy"
    ]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.title(
    "Random Forest Confusion Matrix"
)

plt.tight_layout()

plt.show()


# ============================================================
# 13. ROC-AUC
# ============================================================

# Probability of class 1 = Healthy
y_probability = rf_model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\nROC-AUC:", round(roc_auc, 4))


# ============================================================
# 14. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve - Random Forest"
)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 15. FEATURE IMPORTANCE
# ============================================================

importance = pd.Series(
    rf_model.feature_importances_,
    index=features
)

importance = importance.sort_values(
    ascending=False
)

print("\n========================================")
print("       FEATURE IMPORTANCE")
print("========================================")

print(importance)


plt.figure(figsize=(8, 5))

importance.sort_values().plot(
    kind="barh"
)

plt.xlabel("Importance")
plt.ylabel("Water Quality Parameter")

plt.title(
    "Random Forest Feature Importance"
)

plt.tight_layout()

plt.show()


# ============================================================
# 16. SAVE TEST RESULTS
# ============================================================

results.to_csv(
    "GEMStat_Test_Predictions.csv",
    index=False
)

print("\nTest prediction results saved as:")
print("GEMStat_Test_Predictions.csv")


# ============================================================
# 17. SAVE TRAINED RANDOM FOREST MODEL
# ============================================================

joblib.dump(
    rf_model,
    "GEMStat_RandomForest_v2.pkl"
)

print("\n========================================")
print("       MODEL SAVED")
print("========================================")

print("Saved model:")
print("GEMStat_RandomForest_v2.pkl")

print("\nTraining and testing completed successfully!")