"""The metrics that make this a study of interference rather than a study of
throughput.

Every function here is pure: explicit arguments in, plain dict or dataclass
out, no hidden state. These numbers go into a paper and need to be auditable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class AdapterObservation:
    """Everything measured for one adapter in one configuration."""

    adapter_id: str
    rank: int
    task_score: float
    ttft_s: list[float]
    tpot_ms: list[float]
    num_requests: int
    cache_hits: int = 0
    cache_misses: int = 0
    swap_bytes: float = 0.0

    def p99_ttft(self) -> float:
        return percentile(self.ttft_s, 99.0)

    def p99_tpot(self) -> float:
        return percentile(self.tpot_ms, 99.0)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def interference_degradation_score(
    co_served_score: float, solo_score: float
) -> float:
    """IDS = 1 - score(co-served) / score(solo). Higher is worse.

    A solo score of zero means the adapter never worked in the first place,
    so interference is undefined rather than zero. Returning NaN here is
    deliberate: it surfaces a broken adapter instead of hiding it as a
    perfect result.
    """
    if solo_score == 0.0:
        return float("nan")
    return 1.0 - (co_served_score / solo_score)


def tail_slowdown(co_served_p99: float, solo_p99: float) -> float:
    """Ratio of co-served p99 to solo p99. 1.0 means no resource
    interference."""
    if solo_p99 == 0.0:
        return float("nan")
    return co_served_p99 / solo_p99


def slo_attainment(values: list[float], target: float) -> float:
    """Fraction of requests meeting a latency target, as a percentage."""
    if not values:
        return float("nan")
    met = sum(1 for v in values if v <= target)
    return 100.0 * met / len(values)


def jains_index(service: list[float]) -> float:
    """Jain's fairness index over per-adapter normalised service.

    1.0 is perfectly fair; 1/n is maximally unfair.
    """
    arr = np.asarray(service, dtype=float)
    if arr.size == 0 or not np.any(arr):
        return float("nan")
    return float(arr.sum() ** 2 / (arr.size * np.square(arr).sum()))


def max_pairwise_service_difference(service: list[float]) -> float:
    """Largest gap in normalised service between any two adapters, in the
    spirit of the VTC fairness bound for LLM serving."""
    if len(service) < 2:
        return 0.0
    return float(max(service) - min(service))


def answer_drift(
    predictions_a: list[str], predictions_b: list[str]
) -> float:
    """Fraction of prompts whose greedy decode differs between two runs that
    differ only in batch composition.

    This isolates hypothesis H3 (batched reduction order perturbs logits)
    from every policy effect. Non-determinism is the signal here, not noise,
    so do not seed it away.
    """
    if not predictions_a:
        return float("nan")
    if len(predictions_a) != len(predictions_b):
        raise ValueError(
            "answer_drift needs aligned prediction lists: "
            f"got {len(predictions_a)} and {len(predictions_b)}"
        )
    differing = sum(1 for a, b in zip(predictions_a, predictions_b) if a != b)
    return 100.0 * differing / len(predictions_a)


def summarise_configuration(
    co_served: dict[str, AdapterObservation],
    solo: dict[str, AdapterObservation],
    slo_ttft_s: float,
    slo_tpot_ms: float,
) -> dict[str, Any]:
    """Collapse one configuration into the row that goes in the results CSV.

    Reports mean AND worst-adapter for every interference quantity. A policy
    can look excellent on the mean while destroying a single tenant, which is
    exactly the gap this project exists to close.
    """
    rows: list[dict[str, Any]] = []
    for adapter_id, obs in sorted(co_served.items()):
        if adapter_id not in solo:
            raise KeyError(
                f"no solo reference for adapter {adapter_id!r}; "
                "run the N=1 configuration first"
            )
        ref = solo[adapter_id]
        rows.append(
            {
                "adapter_id": adapter_id,
                "rank": obs.rank,
                "task_score": obs.task_score,
                "solo_task_score": ref.task_score,
                "ids": interference_degradation_score(obs.task_score, ref.task_score),
                "p99_ttft_s": obs.p99_ttft(),
                "ttft_slowdown": tail_slowdown(obs.p99_ttft(), ref.p99_ttft()),
                "tpot_slowdown": tail_slowdown(obs.p99_tpot(), ref.p99_tpot()),
                "slo_ttft_pct": slo_attainment(obs.ttft_s, slo_ttft_s),
                "slo_tpot_pct": slo_attainment(obs.tpot_ms, slo_tpot_ms),
                "cache_hit_rate": _hit_rate(obs),
                "swap_bytes": obs.swap_bytes,
            }
        )

    ids_values = [r["ids"] for r in rows]
    service = [float(co_served[r["adapter_id"]].num_requests) for r in rows]
    total_service = sum(service) or 1.0
    normalised = [s / total_service for s in service]

    summary = {
        "mean_task_score": _nanmean([r["task_score"] for r in rows]),
        "worst_task_score": _nanmin([r["task_score"] for r in rows]),
        "mean_ids": _nanmean(ids_values),
        "worst_ids": _nanmax(ids_values),
        "mean_p99_ttft_s": _nanmean([r["p99_ttft_s"] for r in rows]),
        "worst_ttft_slowdown": _nanmax([r["ttft_slowdown"] for r in rows]),
        "mean_slo_ttft_pct": _nanmean([r["slo_ttft_pct"] for r in rows]),
        "jains_index": jains_index(normalised),
        "max_service_diff": max_pairwise_service_difference(normalised),
    }
    return {"summary": summary, "per_adapter": rows}


def _hit_rate(obs: AdapterObservation) -> float:
    total = obs.cache_hits + obs.cache_misses
    if total == 0:
        return float("nan")
    return 100.0 * obs.cache_hits / total


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def _nanmax(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmax(arr)) if arr.size else float("nan")


def _nanmin(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmin(arr)) if arr.size else float("nan")


def observation_to_dict(obs: AdapterObservation) -> dict[str, Any]:
    """Flatten an observation for CSV writing, dropping the raw latency
    traces but keeping their percentiles."""
    data = asdict(obs)
    data.pop("ttft_s")
    data.pop("tpot_ms")
    data["p99_ttft_s"] = obs.p99_ttft()
    data["p99_tpot_ms"] = obs.p99_tpot()
    return data
