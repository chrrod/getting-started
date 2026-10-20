import numpy as np
import pandas as pd
import aleatoric_features as af
import json

from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# 1. Define your feature names for clarity
ALEATORIC_TEMPORAL_FEATURES: tuple[str, ...] = (
    "early_stage_aleatoric_mean",
    "middle_stage_aleatoric_mean",
    "late_stage_aleatoric_mean",
    "aleatoric_slope",
    "aleatoric_r2_linearity",
)

# --- MOCK DATA FOR DEMONSTRATION ---
# Replace this with your actual DataFrame loading logic (e.g., pd.read_csv("data.csv"))
data_list = []

files = [
    "data/generated/deepseek_test.jsonl",
    "data/generated/olmo.jsonl",
    "data/generated/qwen3-5.jsonl",
    "data/generated/skywork.jsonl",
]

for file_path in files:
    with open(file_path, "r", encoding="utf-8") as f:
        data_list.append([json.loads(line) for line in f])

# Extract features from every dataset
dfs = [
    af.extract_aleatoric_features_dataset(data, temporal_only=True)
    for data in data_list
]

df = pd.concat(dfs, ignore_index=True)

# 3. Separate features and target
X = df[list(ALEATORIC_TEMPORAL_FEATURES)]
y = df["model_is_robust"].astype(int)

# IMPORTANT:
# Replace "group_id" with the actual column identifying
# samples that must stay together in the same fold.
groups = df["model_id"]

# 4. Define the model pipeline
clf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        C=1.0,
        l1_ratio=0,
        class_weight="balanced",
        max_iter=5000,
    )),
])

# 5. Set up 5-fold stratified group cross-validation
cv = StratifiedGroupKFold(
    n_splits=4,
    shuffle=True,
    random_state=464,
)

# 6. Evaluate using cross-validation
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

cv_results = cross_validate(
    estimator=clf_pipeline,
    X=X,
    y=y,
    groups=groups,
    cv=cv,
    scoring=scoring,
    return_train_score=False,
    n_jobs=-1,
)

# 7. Report results for each fold and overall
for metric in scoring:
    scores = cv_results[f"test_{metric}"]

    print(f"\n{metric.upper()}:")
    print(f"  Fold scores: {np.round(scores, 4)}")
    print(f"  Mean:        {scores.mean():.4f}")
    print(f"  Std:         {scores.std():.4f}")