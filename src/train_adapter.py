"""Train one LoRA adapter on the shared frozen base.

The base model is never modified. Every adapter in the zoo is trained against
the same frozen checkpoint, which is what makes the serving-time comparison
meaningful: any difference observed later is interference, not a different
base.

Usage:
    python -m src.train_adapter --config configs/default.yaml \
        --adapter-id a00_reasoning_r4
    python -m src.train_adapter --config configs/default.yaml --all
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import yaml

from .adapters import AdapterSpec, build_zoo
from .data import Example, load_task


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def train_one(spec: AdapterSpec, config: dict[str, Any]) -> str:
    """Train a single adapter and return the directory it was saved to."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    base_cfg = config["base_model"]
    train_cfg = config["train"]
    out_dir = os.path.join(
        config["run"]["artifacts_dir"], "adapters", spec.adapter_id
    )
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_cfg["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_cfg["name"],
        torch_dtype=getattr(torch, base_cfg["dtype"]),
        device_map="auto",
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=spec.rank,
        lora_alpha=spec.lora_alpha,
        lora_dropout=spec.lora_dropout,
        target_modules=spec.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    examples = load_task(
        spec.dataset, split="train", max_samples=int(train_cfg["max_train_samples"])
    )
    dataset = _tokenise(examples, tokenizer, int(train_cfg["max_seq_len"]))

    args = TrainingArguments(
        output_dir=os.path.join(out_dir, "_trainer"),
        num_train_epochs=float(train_cfg["epochs"]),
        per_device_train_batch_size=int(train_cfg["batch_size"]),
        gradient_accumulation_steps=int(train_cfg["grad_accum"]),
        learning_rate=float(train_cfg["learning_rate"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        seed=int(config["run"]["seed"]),
        bf16=base_cfg["dtype"] == "bfloat16",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    with open(os.path.join(out_dir, "zoo_spec.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "adapter_id": spec.adapter_id,
                "family": spec.family,
                "dataset": spec.dataset,
                "metric": spec.metric,
                "rank": spec.rank,
                "lora_alpha": spec.lora_alpha,
                "base_model": base_cfg["name"],
            },
            handle,
            indent=2,
        )
    return out_dir


def _tokenise(examples: list[Example], tokenizer: Any, max_seq_len: int) -> Any:
    from datasets import Dataset

    texts = [f"{ex.prompt} {ex.target}{tokenizer.eos_token}" for ex in examples]
    dataset = Dataset.from_dict({"text": texts})

    def encode(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )

    return dataset.map(encode, batched=True, remove_columns=["text"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--adapter-id", default=None)
    parser.add_argument("--all", action="store_true", help="train the whole zoo")
    args = parser.parse_args()

    config = load_config(args.config)
    zoo = build_zoo(config)

    if args.all:
        targets = zoo
    elif args.adapter_id:
        targets = [s for s in zoo if s.adapter_id == args.adapter_id]
        if not targets:
            available = ", ".join(s.adapter_id for s in zoo)
            raise SystemExit(
                f"adapter {args.adapter_id!r} is not in the zoo. Available: {available}"
            )
    else:
        raise SystemExit("pass --adapter-id <id> or --all")

    for index, spec in enumerate(targets, start=1):
        print(f"[{index}/{len(targets)}] training {spec.adapter_id} (r={spec.rank})")
        path = train_one(spec, config)
        print(f"    saved to {path}")


if __name__ == "__main__":
    main()
