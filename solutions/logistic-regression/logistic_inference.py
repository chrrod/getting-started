"""Offline trace generation and logistic-regression inference for Codabench."""

from __future__ import annotations

import gc
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from logistic_features import FEATURE_NAMES, extract_features

ARTIFACT_PATH = Path(__file__).resolve().parent / "model_artifacts" / "logistic_model.joblib"
MAX_PROMPT_TOKENS = 2_048
MAX_NEW_TOKENS = 4_096


class UncertaintyCollector(LogitsProcessor):
    """Collect the same greedy token statistics used by the training traces."""

    def __init__(self) -> None:
        self.entropy: list[float] = []
        self.top1_probability: list[float] = []
        self.top2_margin: list[float] = []
        self.selected_log_probability: list[float] = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        logits = scores[0].float()
        log_probabilities = torch.log_softmax(logits, dim=-1)
        probabilities = log_probabilities.exp()
        top2_probabilities, top2_ids = torch.topk(probabilities, k=2)
        selected_id = top2_ids[0]

        self.entropy.append(float((-(probabilities * log_probabilities).sum()).cpu()))
        self.top1_probability.append(float(top2_probabilities[0].cpu()))
        self.top2_margin.append(
            float((top2_probabilities[0] - top2_probabilities[1]).cpu())
        )
        self.selected_log_probability.append(float(log_probabilities[selected_id].cpu()))
        return scores


def _format_problem(tokenizer: Any, problem: str, reasoning_effort: str) -> str:
    messages = [{"role": "user", "content": problem}]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (TypeError, ValueError):
            pass
    return problem


def _load_model_and_tokenizer(model_id: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError(f"Tokenizer for {model_id!r} has no pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype="auto",
        local_files_only=True,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def _extract_problem_features(
    model: Any,
    tokenizer: Any,
    problem: str,
    reasoning_effort: str,
) -> dict[str, float]:
    prompt = _format_problem(tokenizer, problem, reasoning_effort)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_TOKENS,
    )
    input_device = model.get_input_embeddings().weight.device
    inputs = {name: value.to(input_device) for name, value in inputs.items()}
    prompt_length = inputs["input_ids"].shape[1]
    collector = UncertaintyCollector()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            logits_processor=LogitsProcessorList([collector]),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output[0, prompt_length:]
    reasoning_trace = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return extract_features(
        selected_log_probs=collector.selected_log_probability,
        entropy=collector.entropy,
        top1_probabilities=collector.top1_probability,
        top2_margins=collector.top2_margin,
        reasoning_trace=reasoning_trace,
    )


def predict_robustness(
    *,
    model_id: str,
    reasoning_effort: str,
    problem_texts: Sequence[str],
) -> list[bool]:
    """Generate traces, calculate features, and return classifier decisions."""

    problems = list(problem_texts)
    if not problems:
        return []
    if not ARTIFACT_PATH.is_file():
        return [False] * len(problems)

    estimator = joblib.load(ARTIFACT_PATH)
    model: Any | None = None
    tokenizer: Any | None = None
    try:
        model, tokenizer = _load_model_and_tokenizer(model_id)
        rows = [
            _extract_problem_features(
                model,
                tokenizer,
                problem,
                reasoning_effort,
            )
            for problem in problems
        ]
        features = pd.DataFrame.from_records(rows, columns=list(FEATURE_NAMES))
        predictions = np.asarray(estimator.predict(features), dtype=int)
        if predictions.shape != (len(problems),):
            raise RuntimeError(
                f"Classifier returned shape {predictions.shape}; expected ({len(problems)},)"
            )
        return [bool(prediction) for prediction in predictions]
    finally:
        model = None
        tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
