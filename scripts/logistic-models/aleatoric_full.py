import numpy as np
import pandas as pd
import aleatoric_features as af
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score


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

ALL_ALEATORIC_FEATURES: tuple[str, ...] = (
    ALEATORIC_AGGREGATE_FEATURES + ALEATORIC_TEMPORAL_FEATURES
)

# --- MOCK DATA FOR DEMONSTRATION ---
# Replace this with your actual DataFrame loading logic (e.g., pd.read_csv("data.csv"))
data_list = []

with open('data/generated/deepseek_test.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]
    data_list.append(data)

with open('data/generated/olmo.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]
    data_list.append(data)

with open('data/generated/qwen3-5.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]
    data_list.append(data)
with open('data/generated/skywork.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]
    data_list.append(data)

df = af.extract_aleatoric_features_dataset(data_list[0])
for i in range(1, len(data_list)):
    df = pd.concat([df, af.extract_aleatoric_features_dataset(data_list[1])], ignore_index=True)

# df = pd.DataFrame(X_data, columns=ALEATORIC_AGGREGATE_FEATURES)
# df['target'] = X_data['model_is_robust']
# # ------------------------------------

# 2. Split features and target
X = df[list(ALL_ALEATORIC_FEATURES)]
y = df['model_is_robust'].astype(int) # Convert boolean True/False to 1/0

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=464, stratify=y
)

# 4. Create a Pipeline with a Scaler and Logistic Regression
# solver='liblinear' or 'lbfgs' work well for tabular datasets
clf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(C=1.0, l1_ratio=0, class_weight = 'balanced', random_state=464))
])

# 5. Train the model
clf_pipeline.fit(X_train, y_train)

# 6. Evaluate the model
y_pred = clf_pipeline.predict(X_test)
y_prob = clf_pipeline.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# Optional: Inspect feature coefficients to see which metrics drive predictions most
model = clf_pipeline.named_steps['classifier']
feature_importance = pd.DataFrame({
    'Feature': ALL_ALEATORIC_FEATURES,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\nTop Feature Coefficients:")
print(feature_importance.head(10))