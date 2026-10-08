import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)


# ---------------------------------------------------------
# Collect token-level uncertainty information during generation
# ---------------------------------------------------------

class UncertaintyCollector(LogitsProcessor):
    def __init__(self):
        self.entropy = []
        self.top1_prob = []
        self.top2_margin = []
        self.selected_logprob = []

    def __call__(self, input_ids, scores):
        # This script assumes batch size = 1
        logits = scores[0].float()

        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Entropy of the next-token probability distribution
        entropy = -(probs * log_probs).sum()

        # Top two token probabilities
        top2_probs, top2_ids = torch.topk(probs, k=2)

        # Since generation is greedy, top-1 is the selected token
        selected_id = top2_ids[0]
        selected_logprob = log_probs[selected_id]

        self.entropy.append(float(entropy.cpu()))
        self.top1_prob.append(float(top2_probs[0].cpu()))
        self.top2_margin.append(
            float((top2_probs[0] - top2_probs[1]).cpu())
        )
        self.selected_logprob.append(float(selected_logprob.cpu()))

        return scores


# ---------------------------------------------------------
# Download/load official AIMO training data
# ---------------------------------------------------------

def load_aimo_data(cache_path):
    cache_path = Path(cache_path)

    if cache_path.exists():
        print(f"Loading cached dataset from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Downloading aimo-interp/train-main-v2...")

    dataset = load_dataset(
        "aimo-interp/train-main-v2",
        split="train"
    )

    df = dataset.to_pandas()

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_parquet(
        cache_path,
        index=False
    )

    print(f"Saved dataset to {cache_path}")

    return df


# ---------------------------------------------------------
# Format the problem for the model
# ---------------------------------------------------------

def format_problem(tokenizer, problem, reasoning_effort):
    messages = [
        {
            "role": "user",
            "content": problem,
        }
    ]

    # Use the model's own chat template when possible
    if tokenizer.chat_template:

        # reasoning_effort may matter for models such as GPT-OSS.
        # Different model templates may expose this differently,
        # so we preserve the value even if the template ignores it.
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return problem


# ---------------------------------------------------------
# Generate one reasoning trace
# ---------------------------------------------------------

def generate_trace(
    model,
    tokenizer,
    problem,
    reasoning_effort,
    max_prompt_tokens,
    max_new_tokens,
):

    prompt = format_problem(
        tokenizer,
        problem,
        reasoning_effort,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
    )

    # Put input tensors on the model's input device
    device = model.get_input_embeddings().weight.device

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    prompt_length = inputs["input_ids"].shape[1]

    collector = UncertaintyCollector()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            logits_processor=LogitsProcessorList(
                [collector]
            ),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[
        0,
        prompt_length:
    ]

    reasoning_trace = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return {
        "reasoning_trace": reasoning_trace,

        "generation_num_tokens":
            int(len(generated_ids)),

        "generated_token_ids":
            generated_ids.cpu().tolist(),

        "entropy_trace":
            collector.entropy,

        "top1_prob_trace":
            collector.top1_prob,

        "top2_margin_trace":
            collector.top2_margin,

        "selected_logprob_trace":
            collector.selected_logprob,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-id",
        required=True,
        help=(
            "Exact model_id from the AIMO dataset, "
            "for example deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
        ),
    )

    parser.add_argument(
        "--model-path",
        required=True,
        help="Local path to the downloaded Hugging Face model",
    )

    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help=(
            "Optional reasoning effort: default, low, medium, or high"
        ),
    )

    parser.add_argument(
        "--output",
        default="data/generated/traces.jsonl",
    )

    parser.add_argument(
        "--dataset-cache",
        default="data/train-main-v2.parquet",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Use only the first N matching examples",
    )

    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Load official AIMO dataset
    # -----------------------------------------------------

    df = load_aimo_data(
        args.dataset_cache
    )

    required_columns = {
        "model_id",
        "reasoning_effort",
        "problem",
        "is_robust",
        "max_drop",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing columns: {missing}"
        )

    # -----------------------------------------------------
    # Keep only labeled examples
    # -----------------------------------------------------

    df = df[
        df["is_robust"].notna()
    ].copy()

    # -----------------------------------------------------
    # Keep only rows for this model
    # -----------------------------------------------------

    df = df[
        df["model_id"] == args.model_id
    ].copy()

    # -----------------------------------------------------
    # Optional reasoning-effort filter
    # -----------------------------------------------------

    if args.reasoning_effort is not None:
        df = df[
            df["reasoning_effort"]
            == args.reasoning_effort
        ].copy()

    if args.max_samples is not None:
        df = df.head(
            args.max_samples
        )

    print(
        f"Found {len(df)} labeled rows "
        f"for {args.model_id}"
    )

    if len(df) == 0:
        raise ValueError(
            "No matching labeled rows were found. "
            "Check model_id and reasoning_effort."
        )

    # -----------------------------------------------------
    # Load tokenizer
    # -----------------------------------------------------

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype="auto",
        local_files_only=True,
        trust_remote_code=True,
    )

    model.eval()

    # -----------------------------------------------------
    # Prepare output
    # -----------------------------------------------------

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Generate traces
    # -----------------------------------------------------

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as fout:

        for row_number, (_, row) in enumerate(
            df.iterrows(),
            start=1
        ):

            print(
                f"[{row_number}/{len(df)}] "
                f"Generating trace..."
            )

            try:

                generated = generate_trace(
                    model=model,
                    tokenizer=tokenizer,
                    problem=row["problem"],
                    reasoning_effort=row["reasoning_effort"],
                    max_prompt_tokens=args.max_prompt_tokens,
                    max_new_tokens=args.max_new_tokens,
                )

                result = {
                    "model_id":
                        row["model_id"],

                    "reasoning_effort":
                        row["reasoning_effort"],

                    "original_problem":
                        row["problem"],

                    "model_is_robust":
                        bool(row["is_robust"]),

                    "max_drop":
                        float(row["max_drop"]),

                    **generated,
                }

                fout.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                # Save after each example so progress
                # is not lost if the process stops
                fout.flush()

            except Exception as error:

                print(
                    f"FAILED example "
                    f"{row_number}: {error}"
                )

    print(
        f"\nFinished. Output saved to:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()