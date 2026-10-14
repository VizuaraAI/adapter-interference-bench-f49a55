"""Dataset loading and per-task metric functions.

Every loader pulls a public dataset from the Hugging Face Hub on first use.
Nothing large is committed to this repository.

`datasets` and the heavier metric libraries are imported lazily inside the
functions that need them, so that `python -m src.measure --simulate` runs on
a machine with only the core requirements installed.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


# Hub coordinates for each dataset key used in configs/default.yaml.
# (hub_name, config_name, split_for_train, split_for_eval)
DATASET_REGISTRY: dict[str, tuple[str, str | None, str, str]] = {
    "gsm8k": ("openai/gsm8k", "main", "train", "test"),
    "mbpp": ("google-research-datasets/mbpp", "full", "train", "test"),
    "spider": ("xlangai/spider", None, "train", "validation"),
    "pubmed_qa": ("qiaojin/PubMedQA", "pqa_labeled", "train", "train"),
    "case_hold": ("coastalcph/lex_glue", "case_hold", "train", "validation"),
    "samsum": ("knkarthick/samsum", None, "train", "test"),
}

# How to turn one raw record into a (prompt, target) pair.
PROMPT_TEMPLATES: dict[str, tuple[str, str]] = {
    "gsm8k": ("question", "answer"),
    "mbpp": ("text", "code"),
    "spider": ("question", "query"),
    "pubmed_qa": ("question", "final_decision"),
    "case_hold": ("context", "label"),
    "samsum": ("dialogue", "summary"),
}

INSTRUCTIONS: dict[str, str] = {
    "gsm8k": "Solve the grade school math problem. End with the final number.",
    "mbpp": "Write a Python function that satisfies the description.",
    "spider": "Translate the question into a single SQL query.",
    "pubmed_qa": "Answer the biomedical question with yes, no, or maybe.",
    "case_hold": "Choose the holding that correctly completes the citation.",
    "samsum": "Summarise the dialogue in two or three sentences.",
}


@dataclass
class Example:
    prompt: str
    target: str


def load_task(
    dataset_key: str, split: str = "eval", max_samples: int | None = None
) -> list[Example]:
    """Load one task as a list of (prompt, target) examples.

    `split` is either "train" or "eval"; the concrete split name per dataset
    lives in DATASET_REGISTRY so the caller does not need to know it.
    """
    if dataset_key not in DATASET_REGISTRY:
        raise ValueError(
            f"unknown dataset {dataset_key!r}; "
            f"expected one of {sorted(DATASET_REGISTRY)}"
        )
    from datasets import load_dataset  # lazy: keeps --simulate dependency-light

    hub_name, config_name, train_split, eval_split = DATASET_REGISTRY[dataset_key]
    split_name = train_split if split == "train" else eval_split

    raw = load_dataset(hub_name, config_name, split=split_name)
    if max_samples is not None:
        raw = raw.select(range(min(max_samples, len(raw))))

    prompt_field, target_field = PROMPT_TEMPLATES[dataset_key]
    instruction = INSTRUCTIONS[dataset_key]

    examples: list[Example] = []
    for record in raw:
        prompt = f"{instruction}\n\n{record[prompt_field]}\n\nAnswer:"
        examples.append(Example(prompt=prompt, target=str(record[target_field])))
    return examples


# ---------------------------------------------------------------------------
# Metrics. Each returns a score in [0, 100] so that IDS is a clean ratio.
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def exact_match(predictions: list[str], targets: list[str]) -> float:
    """Exact match on the final numeric answer, for math-style tasks."""
    if not predictions:
        return 0.0
    hits = 0
    for pred, target in zip(predictions, targets):
        pred_nums = re.findall(r"-?\d+\.?\d*", pred.replace(",", ""))
        target_nums = re.findall(r"-?\d+\.?\d*", target.replace(",", ""))
        if pred_nums and target_nums and pred_nums[-1] == target_nums[-1]:
            hits += 1
    return 100.0 * hits / len(predictions)


def accuracy(predictions: list[str], targets: list[str]) -> float:
    """String accuracy after normalisation, for classification-style tasks."""
    if not predictions:
        return 0.0
    hits = sum(
        1
        for pred, target in zip(predictions, targets)
        if _normalise(target) in _normalise(pred)
    )
    return 100.0 * hits / len(predictions)


def pass_at_1(predictions: list[str], targets: list[str]) -> float:
    """Execution-free proxy for pass@1.

    This is a structural check only: it rewards syntactically valid Python
    that defines a function. Replace it with sandboxed execution against the
    dataset's test cases before you report pass@1 in the paper. The proxy is
    here so the pipeline runs end to end on day one.
    """
    if not predictions:
        return 0.0
    import ast

    hits = 0
    for pred in predictions:
        try:
            tree = ast.parse(pred)
        except SyntaxError:
            continue
        if any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree)):
            hits += 1
    return 100.0 * hits / len(predictions)


def rouge_l(predictions: list[str], targets: list[str]) -> float:
    """ROUGE-L F-measure, for summarisation tasks."""
    if not predictions:
        return 0.0
    from rouge_score import rouge_scorer  # lazy import

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    total = sum(
        scorer.score(target, pred)["rougeL"].fmeasure
        for pred, target in zip(predictions, targets)
    )
    return 100.0 * total / len(predictions)


METRIC_FUNCTIONS: dict[str, Callable[[list[str], list[str]], float]] = {
    "exact_match": exact_match,
    "accuracy": accuracy,
    "pass_at_1": pass_at_1,
    "rouge_l": rouge_l,
}


def score(metric: str, predictions: list[str], targets: list[str]) -> float:
    if metric not in METRIC_FUNCTIONS:
        raise ValueError(
            f"unknown metric {metric!r}; expected one of {sorted(METRIC_FUNCTIONS)}"
        )
    return METRIC_FUNCTIONS[metric](predictions, targets)


def describe_registry() -> dict[str, Any]:
    """Small helper used by scripts to print what will be downloaded."""
    return {
        key: {"hub": value[0], "config": value[1]}
        for key, value in DATASET_REGISTRY.items()
    }
