# AIMO Uncertainty Benchmark Datasets

This directory contains standardized, curated benchmark datasets designed for the **AIMO Interpretability Challenge**. These datasets evaluate whether **temporal dynamics** (reasoning evolution over generation steps) and **epistemic uncertainty** (model-level doubt vs. token-level data entropy) can predict whether an LLM will solve an Olympiad-level mathematics problem **robustly** or **spuriously**.

---

## 1. Overview of the Benchmark Datasets

All datasets are derived from identical problem-model reasoning traces generated across four representative competition model families:
- `Qwen/Qwen3.5-4B`
- `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`
- `allenai/Olmo-3-7B-Think`
- `Skywork/Skywork-OR1-Math-7B`

Every dataset strictly preserves:
1. **Identical row count**: Exactly 34 traces in the base evaluation suite.
2. **Canonical row ordering**: Deterministically sorted by `(problem_id, model_id)`.
3. **Core metadata & target parity**: Exactly matching `problem_id`, `model_id`, `generation_num_tokens`, `original_problem`, `reasoning_trace`, and `model_is_robust`.
4. **Zero Missing Data**: 0 `NaN`, `Inf`, or null entries across all feature columns.

| Dataset File | Name | Uncertainty Type | Granularity | Core Cols | Feature Cols | Total Cols | Purpose |
|---|---|---|---|---|---|---|---|
| `dataset_1_aleatoric_aggregate.parquet` | Aleatoric Aggregate | Aleatoric (Data-Based) | Aggregate | 6 | 14 | 20 | Static baseline assessing overall next-token confidence |
| `dataset_2_aleatoric_temporal.parquet` | Aleatoric Temporal | Aleatoric (Data-Based) | Temporal | 6 | 5 | 11 | Evaluates whether the trajectory of token entropy over time predicts robustness |
| `dataset_3_epistemic_aggregate.parquet` | Epistemic Aggregate | Epistemic (Model-Based) | Aggregate | 6 | 5 | 11 | Static model-level uncertainty (dispersion across weights/layers) |
| `dataset_4_epistemic_temporal.parquet` | Epistemic Temporal | Epistemic (Model-Based) | Temporal | 6 | 5 | 11 | Evaluates whether shifts in epistemic doubt across reasoning stages signal errors |
| `dataset_5_multimodal_combined.parquet` | Multimodal Combined | Multimodal (Both) | Combined | 6 | 29 | 35 | Full 29-feature multimodal system for ensemble modeling |
| `dataset_aleatoric_full.parquet` | Aleatoric Full *(Ablation)* | Aleatoric | Full (Agg + Temp) | 6 | 19 | 25 | Evaluates combined static + temporal aleatoric predictive power |
| `dataset_epistemic_full.parquet` | Epistemic Full *(Ablation)* | Epistemic | Full (Agg + Temp) | 6 | 10 | 16 | Evaluates combined static + temporal epistemic predictive power |

---

## 2. Core Metadata & Target Schema (6 Columns)

Every dataset includes these 6 foundational columns as its leading fields:

| Column Name | Type | Description |
|---|---|---|
| `problem_id` | `str` | Deterministic SHA-256 hash of the problem text (e.g. `prob_37172384f0dc`), guaranteed identical across shared prompts. |
| `model_id` | `str` | Hugging Face repository identifier for the generating model. |
| `generation_num_tokens` | `int64` | Total number of tokens generated in the reasoning chain $T$. |
| `original_problem` | `str` | Verbatim text of the mathematical problem prompt. |
| `reasoning_trace` | `str` | Full generated chain-of-thought text emitted by the model. |
| `model_is_robust` | `bool` | **Ground truth target label**: `True` if solution is mathematically robust under perturbation; `False` if spurious or fragile. |

### Target Class Distribution
- **Robust (`True`)**: 24 instances (70.6%)
- **Spurious (`False`)**: 10 instances (29.4%)
- **Class Balance Ratio**: ~2.4:1

---

## 3. Uncertainty Feature Dictionary (29 Features Total)

### 3.1. Aleatoric Aggregate Features (14 Features) — Dataset 1
These capture token-level unpredictability and token distribution sharpness across the full trace $t = 1 \dots T$:

| Feature Name | Formulation / Definition | Intuition & Range |
|---|---|---|
| `mean_entropy` | $\frac{1}{T} \sum_{t=1}^T \mathcal{H}(P(y_t \mid y_{<t}))$ | Average Shannon entropy of next-token distribution. Higher = diffuse choices. $[0, \infty)$ |
| `mean_token_probability` | $\frac{1}{T} \sum_{t=1}^T P(y_t \mid y_{<t})$ | Average probability mass assigned to the chosen token. Higher = confident. $[0, 1]$ |
| `mean_log_probability` | $\frac{1}{T} \sum_{t=1}^T \log P(y_t \mid y_{<t})$ | Mean log probability across generation. $(-\infty, 0]$ |
| `mean_negative_log_likelihood` | $-\frac{1}{T} \sum_{t=1}^T \log P(y_t \mid y_{<t})$ | Average surprisal / cross-entropy loss per token. Higher = more surprised. $[0, \infty)$ |
| `perplexity` | $\exp(\text{mean\_nll})$ | Exponentiated surprisal. Represents effective branching factor. $[1, \infty)$ |
| `min_log_probability` | $\min_{t} \log P(y_t \mid y_{<t})$ | Lowest token confidence encountered (the single biggest "stumble"). $(-\infty, 0]$ |
| `max_log_probability` | $\max_{t} \log P(y_t \mid y_{<t})$ | Highest single-token confidence. $(-\infty, 0]$ |
| `std_log_probability` | $\sqrt{\text{Var}(\log P(y_t \mid y_{<t}))}$ | Standard deviation of log-probabilities across steps. Higher = uneven confidence. $[0, \infty)$ |
| `min_k_log_probability` | Mean of bottom $10\%$ lowest log probabilities | Quantifies average severity of the trace's most uncertain 10% tokens. $(-\infty, 0]$ |
| `mean_top1_probability` | $\frac{1}{T} \sum_{t=1}^T \max_k P(v_k \mid y_{<t})$ | Mean probability assigned to the mode token. Higher = peaked distribution. $[0, 1]$ |
| `mean_top2_probability_margin` | $\frac{1}{T} \sum_{t=1}^T (p_{\text{top1}} - p_{\text{top2}})$ | Mean margin between top-1 and top-2 candidates. Small margin = acute hesitation. $[0, 1]$ |
| `fraction_selected_is_top1` | $\frac{1}{T} \sum_{t=1}^T \mathbf{1}(y_t = \text{argmax} P)$ | Frequency that the selected token was greedy mode ($1.0$ under greedy decoding). $[0, 1]$ |
| `fraction_high_confidence` | $\frac{1}{T} \sum_{t=1}^T \mathbf{1}(P(y_t) \ge 0.90)$ | Percentage of tokens generated with high certainty ($\ge 90\%$). $[0, 1]$ |
| `fraction_low_confidence` | $\frac{1}{T} \sum_{t=1}^T \mathbf{1}(P(y_t) < 0.20)$ | Percentage of tokens generated with low certainty ($< 20\%$). $[0, 1]$ |

---

### 3.2. Aleatoric Temporal Features (5 Features) — Dataset 2
These capture whether next-token uncertainty rises, falls, or fluctuates as the mathematical proof progresses:

| Feature Name | Formulation / Definition | Intuition & Range |
|---|---|---|
| `early_stage_aleatoric_mean` | Mean entropy over first third: $t \in [1, \lfloor T/3 \rfloor]$ | Initial problem comprehension and plan formulation uncertainty. $[0, \infty)$ |
| `middle_stage_aleatoric_mean` | Mean entropy over middle third: $t \in [\lfloor T/3 \rfloor + 1, \lfloor 2T/3 \rfloor]$ | Core calculation and derivation uncertainty. $[0, \infty)$ |
| `late_stage_aleatoric_mean` | Mean entropy over final third: $t \in [\lfloor 2T/3 \rfloor + 1, T]$ | Solution synthesis, sanity checks, and final box output uncertainty. $[0, \infty)$ |
| `aleatoric_slope` | OLS linear regression slope $m$ in $u_t \sim m \cdot \frac{t}{T} + c$ | Rate of uncertainty change. Negative = model gains clarity; positive = model gets confused. $(-\infty, \infty)$ |
| `aleatoric_r2_linearity` | Coefficient of determination $R^2 \in [0, 1]$ of linear trend | Goodness-of-fit. High = steady monotonic trend; low = volatile bursts/spikes. $[0, 1]$ |

---

### 3.3. Epistemic Aggregate Features (5 Features) — Dataset 3
Epistemic uncertainty reflects **model-level doubt** (e.g., lack of knowledge about mathematical lemmas or algorithms), quantified via hidden state dispersion across layers, Monte Carlo dropout, or local token volatility filtering:

| Feature Name | Formulation / Definition | Intuition & Range |
|---|---|---|
| `mean_epistemic_uncertainty` | $\frac{1}{T} \sum_{t=1}^T e_t$ | Average epistemic doubt across the entire reasoning trace. $[0, \infty)$ |
| `min_epistemic_uncertainty` | $\min_t e_t$ | Baseline minimum model uncertainty. $[0, \infty)$ |
| `max_epistemic_uncertainty` | $\max_t e_t$ | Peak model epistemic doubt (point of maximum conceptual ambiguity). $[0, \infty)$ |
| `std_epistemic_uncertainty` | $\sqrt{\text{Var}(e_t)}$ | Variability of epistemic doubt across reasoning steps. $[0, \infty)$ |
| `variance_epistemic_uncertainty` | $\text{Var}(e_t)$ | Variance of epistemic uncertainty. $[0, \infty)$ |

---

### 3.4. Epistemic Temporal Features (5 Features) — Dataset 4
These capture whether internal model doubt accumulates or resolves over time:

| Feature Name | Formulation / Definition | Intuition & Range |
|---|---|---|
| `early_stage_epistemic_mean` | Mean epistemic uncertainty $e_t$ in early phase ($0 - 33\%$) | Model knowledge gaps when reading the problem setup. $[0, \infty)$ |
| `middle_stage_epistemic_mean` | Mean epistemic uncertainty $e_t$ in mid phase ($33 - 66\%$) | Model knowledge gaps during intermediate algebraic steps. $[0, \infty)$ |
| `late_stage_epistemic_mean` | Mean epistemic uncertainty $e_t$ in late phase ($66 - 100\%$) | Model knowledge gaps during conclusion and final answer extraction. $[0, \infty)$ |
| `epistemic_slope` | OLS linear regression slope $m$ of epistemic uncertainty $e_t \sim m \cdot \frac{t}{T} + c$ | Trajectory of model doubt. Negative = doubt resolved; positive = compounding confusion. $(-\infty, \infty)$ |
| `epistemic_r2_linearity` | Coefficient of determination $R^2 \in [0, 1]$ of epistemic linear trend | Consistency of epistemic progression over the normalized timeline. $[0, 1]$ |

---

## 4. Leakage-Free Cross-Validation Protocol

Because multiple reasoning traces may be generated for the same underlying mathematical problem, standard random cross-validation results in severe data leakage.

**Mandatory CV Protocol**:
- Use `StratifiedGroupKFold(n_splits=5)`.
- Group by `problem_id` (guaranteeing that all solutions to the same problem remain either entirely in train or entirely in test).
- Stratify on `model_is_robust` (guaranteeing balanced positive/negative label proportions across folds).

```python
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

df = pd.read_parquet("data/uncertainty_datasets/dataset_1_aleatoric_aggregate.parquet")

sgkf = StratifiedGroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, df["model_is_robust"], groups=df["problem_id"])):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    # Zero problem overlap guaranteed:
    assert len(set(train_df["problem_id"]).intersection(set(val_df["problem_id"]))) == 0
```

---

## 5. Quickstart: Loading Datasets in Python

### Using Pandas
```python
import pandas as pd

# Load individual benchmark datasets
df_aleatoric_agg = pd.read_parquet("data/uncertainty_datasets/dataset_1_aleatoric_aggregate.parquet")
df_aleatoric_temp = pd.read_parquet("data/uncertainty_datasets/dataset_2_aleatoric_temporal.parquet")
df_epistemic_agg = pd.read_parquet("data/uncertainty_datasets/dataset_3_epistemic_aggregate.parquet")
df_epistemic_temp = pd.read_parquet("data/uncertainty_datasets/dataset_4_epistemic_temporal.parquet")
df_combined = pd.read_parquet("data/uncertainty_datasets/dataset_5_multimodal_combined.parquet")

# Verify perfect alignment across all datasets
assert df_aleatoric_agg["problem_id"].equals(df_combined["problem_id"])
assert df_aleatoric_agg["model_is_robust"].equals(df_combined["model_is_robust"])
```

### Re-generating or Verifying Datasets
```bash
# Full curation and verification
uv run python scripts/curate_benchmark_datasets.py

# Verification only
uv run python scripts/curate_benchmark_datasets.py --verify-only
```

