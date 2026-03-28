import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1) Load + clean
DATA_PATH = r"/home/hertz/4thP/DataSet/Disease and symptoms dataset (Copy).csv"
TESTS_DIR = r"./tests"

df = pd.read_csv(DATA_PATH).drop_duplicates().copy()

# 2) Binary target: anxiety vs rest
TARGET_DISEASE = "anxiety"
df["target"] = (df["diseases"] == TARGET_DISEASE).astype(int)

# 3) Split to X, y
X = df.drop(columns=["diseases", "target"])
y = df["target"]

# 4) Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5) Train model
model = DecisionTreeClassifier(class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# 6) Probabilities on test set + ROC-AUC (computed once)
proba = model.predict_proba(X_test)[:, 1]
roc = roc_auc_score(y_test, proba)
print("ROC-AUC:", roc)


# Helper: compute specificity from confusion matrix
def specificity_from_cm(cm):
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp) if (tn + fp) != 0 else 0.0


# 7) Threshold experiment (0.3, 0.5, 0.7)
for threshold in [0.3, 0.5, 0.7]:
    y_pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    print("\nThreshold:", threshold)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall (Sensitivity):", recall_score(y_test, y_pred, zero_division=0))
    print("Specificity:", specificity_from_cm(cm))
    print("F1:", f1_score(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:\n", cm)

fpr, tpr, thresholds = roc_curve(y_test, proba)
auc = roc_auc_score(y_test, proba)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Recall / Sensitivity)")
plt.title("ROC Curve - Anxiety vs Rest")
plt.legend()
plt.grid(True)
plt.show()


# Helper: load + align a custom single input
def load_custom_input(filepath, feature_columns):
    x = pd.read_csv(filepath, sep="\t")

    # If loaded as (377,1) (symptoms as rows), transpose to (1,377)
    if x.shape[0] == len(feature_columns) and x.shape[1] == 1:
        x = x.T

    # Align columns and enforce 0/1 integer
    x = x.reindex(columns=feature_columns, fill_value=0)
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    # Keep only first row if file accidentally contains multiple rows
    x = x.iloc[:1]
    return x


# 8) Test custom input 1 (anxiety_first_row.txt)
X_test2 = load_custom_input(f"{TESTS_DIR}/anxiety_first_row.txt", X.columns)
proba2 = model.predict_proba(X_test2)[:, 1]
p2 = proba2[0] * 100
print(f"\nCustom input (anxiety_first_row) → Probability of Anxiety: {p2:.2f}%")
for threshold in [0.3, 0.5, 0.7]:
    decision = int(proba2[0] >= threshold)
    label = "Anxiety" if decision == 1 else "Not Anxiety"
    print(f"anxiety_first_row @ threshold {threshold}: {label} (decision={decision})")

# 9) Test custom input 2 (nose_first_row.txt)
X_test3 = load_custom_input(f"{TESTS_DIR}/nose_first_row.txt", X.columns)
proba3 = model.predict_proba(X_test3)[:, 1]
p3 = proba3[0] * 100
print(f"\nCustom input (nose_first_row) → Probability of Anxiety: {p3:.2f}%")
for threshold in [0.3, 0.5, 0.7]:
    decision = int(proba3[0] >= threshold)
    label = "Anxiety" if decision == 1 else "Not Anxiety"
    print(f"nose_first_row @ threshold {threshold}: {label} (decision={decision})")

feature_importance = pd.DataFrame(
    {"Symptom": X.columns, "Importance": model.feature_importances_}
)

# Most important symptoms for the tree's decisions
top_positive = feature_importance.sort_values(by="Importance", ascending=False).head(10)

top_positive.set_index("Symptom")["Importance"].plot(kind="barh")
plt.title("Top Important Symptoms (Decision Tree)")
plt.show()

# Least important symptoms
top_negative = feature_importance.sort_values(by="Importance").head(10)
top_negative.set_index("Symptom")["Importance"].plot(kind="barh", color="red")
plt.title("Least Important Symptoms (Decision Tree)")
plt.show()

print("\nTop important symptoms (Decision Tree):")
print(top_positive)

print("\nLeast important symptoms (Decision Tree):")
print(top_negative)
