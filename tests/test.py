import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

DATA_PATH = r"./DataSet/Disease and symptoms dataset (Copy).csv"

df = pd.read_csv(DATA_PATH)
sys = df.columns.drop("diseases")
with open("sys.txt", "w") as f:
    for s in sys:
        f.write(s + "\n")

df.size
df.drop_duplicates(inplace=True)
df.size
pd.set_option("display.max_columns", 5)
df.head()
df = df.drop_duplicates()

all = df["diseases"].value_counts().sum()
counts = df["diseases"].value_counts()["anxiety"]
print(all, counts)

df = df.copy()

df["target"] = (df["diseases"] == "anxiety").astype(int)

X = df.drop(columns=["diseases", "target"])
y = df["target"]
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = LogisticRegression(max_iter=2000, class_weight="balanced")

model.fit(X_train, y_train)

X_test2 = pd.read_csv("./tests/anxiety_first_row.txt", sep="\t", index_col=0)
proba2 = model.predict_proba(X_test2)[:, 1]

proba = model.predict_proba(X_test)[:, 1]
for threshold in [0.3, 0.5, 0.7]:
    y_pred = (proba >= threshold).astype(int)
    p = proba[0] * 100
    print(f"Probability of Anxiety: {p:.2f}%")
    print("\nThreshold:", threshold)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    roc = roc_auc_score(y_test, proba)
    print("ROC-AUC:", roc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
