import numpy as np
import pandas as pd
import aleatoric_features as af
import extractFeatures as ef
import json

from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# 1. Define your feature names for clarity
ALEATORIC_AGGREGATE_FEATURES: tuple[str, ...] = (
    "mean_entropy",
    "mean_token_probability",
    "mean_log_probability",
    "mean_negative_log_likelihood",
    "perplexity",
    "min_log_probability",
    "max_log_probability",
    "std_log_probability",
    "min_k_log_probability",
    "mean_top1_probability",
    "mean_top2_probability_margin",
    "fraction_selected_is_top1",
    "fraction_high_confidence",
    "fraction_low_confidence",
)

ALEATORIC_TEMPORAL_FEATURES: tuple[str, ...] = (
    "early_stage_aleatoric_mean",
    "middle_stage_aleatoric_mean",
    "late_stage_aleatoric_mean",
    "aleatoric_slope",
    "aleatoric_r2_linearity",
)

EPISTEMIC_AGGREGATE_FEATURES: tuple[str, ...] = (
    "mean_epistemic_uncertainty",
    "min_epistemic_uncertainty",
    "max_epistemic_uncertainty",
    "std_epistemic_uncertainty",
    "variance_epistemic_uncertainty",
)

EPISTEMIC_TEMPORAL_FEATURES: tuple[str, ...] = (
    "early_stage_epistemic_mean",
    "middle_stage_epistemic_mean",
    "late_stage_epistemic_mean",
    "epistemic_slope",
    "epistemic_r2_linearity",
)

ALL_ALEATORIC_FEATURES: tuple[str, ...] = (
    ALEATORIC_AGGREGATE_FEATURES + ALEATORIC_TEMPORAL_FEATURES
)

ALL_EPISTEMIC_FEATURES: tuple[str, ...] = (
    EPISTEMIC_AGGREGATE_FEATURES + EPISTEMIC_TEMPORAL_FEATURES
)

ALL_29_UNCERTAINTY_FEATURES: tuple[str, ...] = (
    ALL_ALEATORIC_FEATURES + ALL_EPISTEMIC_FEATURES
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
    ef.extract_features_dataset(data)
    for data in data_list
]

df = pd.concat(dfs, ignore_index=True)

# 3. Separate features and target
X = df[list(ALL_29_UNCERTAINTY_FEATURES)]
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