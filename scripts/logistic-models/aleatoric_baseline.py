import numpy as np
import pandas as pd
import aleatoric_features as af
import json
import joblib

from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ALEATORIC_AGGREGATE_FEATURES = (
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


dfs = [
    af.extract_aleatoric_features_dataset(data, aggregate_only=True)
    for data in data_list
]

df = pd.concat(dfs, ignore_index=True)

X = df[list(ALEATORIC_AGGREGATE_FEATURES)]
y = df["model_is_robust"].astype(int)

groups = df["model_id"]

clf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        C=1.0,
        l1_ratio=0,
        class_weight="balanced",
        max_iter=5000,
    )),
])

cv = StratifiedGroupKFold(
    n_splits=4,
    shuffle=True,
    random_state=464,
)

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
    return_estimator=True,
    n_jobs=-1,
)

for fold, estimator in enumerate(cv_results["estimator"], start=1):
    joblib.dump(
        estimator,
        f"aleatoric_baseline_{fold}.joblib"
    )

for metric in scoring:
    scores = cv_results[f"test_{metric}"]

    print(f"\n{metric.upper()}:")
    print(f"  Fold scores: {np.round(scores, 4)}")
    print(f"  Mean:        {scores.mean():.4f}")
    print(f"  Std:         {scores.std():.4f}")