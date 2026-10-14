"""The sweep driver.

Runs the solo reference for every adapter first, then every co-served
configuration, and writes one append-only CSV row per (configuration,
adapter) pair plus a rendered summary table.

The columns of the output map one-to-one onto the main results table in the
paper draft and onto the rows of experiments/log_template.md.

Usage:
    # fast scaffolding check, no GPU required
    python -m src.measure --config configs/default.yaml --simulate --adapters 4

    # the real sweep
    python -m src.measure --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import random
import time
from typing import Any

import yaml

from .adapters import AdapterSpec, build_zoo, working_set_bytes
from .metrics import AdapterObservation, summarise_configuration
from .policies import (
    AdapterCache,
    InterferenceAwareScheduler,
    make_eviction_policy,
    split_pool,
)
from .serving import (
    FallbackConfig,
    SimulatorBackend,
    VLLMBackend,
    build_workload,
)

# Base model geometry used to size adapters. Overridden from the live model
# config on real runs; these defaults match Qwen2.5-1.5B.
DEFAULT_HIDDEN_SIZE = 1536
DEFAULT_NUM_LAYERS = 28

CSV_COLUMNS = [
    "run_id",
    "backend",
    "concurrency",
    "budget",
    "load_skew",
    "fallback",
    "eviction",
    "adapter_cap",
    "adapter_id",
    "rank",
    "task_score",
    "solo_task_score",
    "ids",
    "p99_ttft_s",
    "ttft_slowdown",
    "tpot_slowdown",
    "slo_ttft_pct",
    "slo_tpot_pct",
    "cache_hit_rate",
    "swap_bytes",
]


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_backend(
    config: dict[str, Any], simulate: bool, solo_scores: dict[str, float]
) -> Any:
    if simulate:
        return SimulatorBackend(config, solo_scores)
    return VLLMBackend(config)


def make_cache(
    specs: list[AdapterSpec],
    config: dict[str, Any],
    budget: float,
    eviction: str,
    dtype: str,
) -> AdapterCache:
    """Build the adapter cache for one configuration.

    The budget is a fraction of the working set's total adapter bytes, then
    the explicit pool split decides how much of the pool the adapter cache
    may actually claim.
    """
    total = working_set_bytes(specs, DEFAULT_HIDDEN_SIZE, DEFAULT_NUM_LAYERS, dtype)
    pool_bytes = total / max(budget, 1e-6)
    adapter_bytes, _kv_bytes = split_pool(
        pool_bytes,
        float(config["policy"]["adapter_cache_floor"]),
        float(config["policy"]["kv_cache_floor"]),
    )
    # The cache must always hold at least the single largest adapter,
    # otherwise no request can ever be served.
    largest = max(
        s.size_bytes(DEFAULT_HIDDEN_SIZE, DEFAULT_NUM_LAYERS, dtype) for s in specs
    )
    capacity = max(min(adapter_bytes, total * budget), largest)

    return AdapterCache(
        capacity_bytes=capacity,
        policy=make_eviction_policy(eviction),
        eviction_weights=dict(config["policy"]["eviction_weights"]),
    )


def run_solo_reference(
    specs: list[AdapterSpec],
    config: dict[str, Any],
    simulate: bool,
    seed: int,
) -> dict[str, AdapterObservation]:
    """Serve each adapter alone. This is the counterfactual every IDS is
    measured against, so it runs first and never shares the machine."""
    solo: dict[str, AdapterObservation] = {}
    # On the simulator path the solo score is a fixed per-adapter constant;
    # on the real path it comes from the adapter's own evaluation set.
    placeholder_scores = {s.adapter_id: 100.0 for s in specs}

    for spec in specs:
        rng = random.Random(seed)
        backend = make_backend(config, simulate, placeholder_scores)
        cache = make_cache([spec], config, budget=1.0, eviction="lru", dtype="bfloat16")
        scheduler = InterferenceAwareScheduler(adapter_concurrency_cap=1)
        requests = build_workload([spec], config, "uniform", rng)
        observations = backend.run(
            [spec], requests, cache, scheduler, FallbackConfig("none")
        )
        solo[spec.adapter_id] = observations[spec.adapter_id]
    return solo


def run_sweep(
    config: dict[str, Any], simulate: bool, limit_adapters: int | None
) -> list[dict[str, Any]]:
    zoo = build_zoo(config)
    if limit_adapters is not None:
        zoo = zoo[:limit_adapters]

    seed = int(config["run"]["seed"])
    run_id = time.strftime("%Y%m%d_%H%M%S")
    sweep = config["sweep"]
    policy_cfg = config["policy"]
    workload_cfg = config["workload"]

    print(f"solo reference for {len(zoo)} adapters")
    solo = run_solo_reference(zoo, config, simulate, seed)
    solo_scores = {k: v.task_score for k, v in solo.items()}

    rows: list[dict[str, Any]] = []
    concurrencies = [n for n in sweep["concurrency"] if 1 < n <= len(zoo)]
    combinations = list(
        itertools.product(
            concurrencies, sweep["budget"], sweep["load_skew"], sweep["fallback"]
        )
    )
    print(f"{len(combinations)} co-served configurations to run")

    for concurrency, budget, skew, fallback_kind in combinations:
        specs = zoo[:concurrency]
        fallback = FallbackConfig(
            kind=fallback_kind,
            truncate_rank_to=int(sweep["truncate_rank_to"]),
            grouped_merge_k=int(sweep["grouped_merge_k"]),
        )
        rng = random.Random(seed)
        backend = make_backend(config, simulate, solo_scores)
        cache = make_cache(
            specs,
            config,
            budget=float(budget),
            eviction=policy_cfg["eviction"],
            dtype=fallback.adapter_dtype(),
        )
        scheduler = InterferenceAwareScheduler(
            adapter_concurrency_cap=int(policy_cfg["adapter_concurrency_cap"])
        )
        requests = build_workload(specs, config, skew, rng)

        observations = backend.run(specs, requests, cache, scheduler, fallback)
        result = summarise_configuration(
            observations,
            {k: solo[k] for k in observations},
            slo_ttft_s=float(workload_cfg["slo_ttft_s"]),
            slo_tpot_ms=float(workload_cfg["slo_tpot_ms"]),
        )

        print(
            f"  N={concurrency:>2} B={budget:<5} skew={skew:<7} "
            f"fallback={fallback_kind:<14} "
            f"mean_ids={result['summary']['mean_ids']:+.4f} "
            f"worst_ids={result['summary']['worst_ids']:+.4f} "
            f"worst_ttft_slowdown={result['summary']['worst_ttft_slowdown']:.2f}x"
        )

        for per_adapter in result["per_adapter"]:
            rows.append(
                {
                    "run_id": run_id,
                    "backend": backend.name,
                    "concurrency": concurrency,
                    "budget": budget,
                    "load_skew": skew,
                    "fallback": fallback_kind,
                    "eviction": policy_cfg["eviction"],
                    "adapter_cap": policy_cfg["adapter_concurrency_cap"],
                    **per_adapter,
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]], path: str) -> None:
    """Append-only. Never overwrite a results file: lost measurements cost
    GPU hours."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def write_table(rows: list[dict[str, Any]], path: str) -> None:
    """Render the markdown summary whose shape matches the paper's table."""
    import pandas as pd

    if not rows:
        print("no rows to tabulate")
        return

    frame = pd.DataFrame(rows)
    grouped = (
        frame.groupby(["concurrency", "budget", "load_skew", "fallback"])
        .agg(
            mean_task_score=("task_score", "mean"),
            worst_task_score=("task_score", "min"),
            mean_ids=("ids", "mean"),
            worst_ids=("ids", "max"),
            p99_ttft_s=("p99_ttft_s", "mean"),
            worst_ttft_slowdown=("ttft_slowdown", "max"),
            mean_slo_ttft_pct=("slo_ttft_pct", "mean"),
        )
        .reset_index()
        .round(4)
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Sweep summary\n\n")
        handle.write(
            "Worst-adapter columns are the ones that matter. A policy can "
            "look excellent on the mean while destroying one tenant.\n\n"
        )
        handle.write(grouped.to_markdown(index=False))
        handle.write("\n")
    print(f"wrote summary table to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="use the analytic backend: no GPU, resource channel only, "
        "IDS is zero by construction",
    )
    parser.add_argument(
        "--adapters",
        type=int,
        default=None,
        help="limit the zoo to the first N adapters (for quick checks)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    rows = run_sweep(config, simulate=args.simulate, limit_adapters=args.adapters)

    out_dir = config["run"]["out_dir"]
    write_csv(rows, os.path.join(out_dir, "sweep.csv"))
    write_table(rows, os.path.join(out_dir, "table.md"))

    if args.simulate:
        print(
            "\nreminder: --simulate models the resource channel only. "
            "Non-zero IDS here means the harness is broken, not that you "
            "found interference."
        )


if __name__ == "__main__":
    main()
