"""The adapter zoo: which adapters exist, what rank each one has, and how
big it is in bytes.

The zoo is deliberately heterogeneous in rank, because rank heterogeneity is
one of the controlled factors of the study: we want to know whether
high-rank, low-traffic tenants absorb a disproportionate share of the damage
under recency-only eviction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Bytes per parameter for a bfloat16 adapter. The fallback ladder changes
# this (8-bit and 4-bit quantisation), which is why size is computed rather
# than hardcoded.
BYTES_PER_PARAM = {"bfloat16": 2.0, "int8": 1.0, "int4": 0.5}


@dataclass
class AdapterSpec:
    """One adapter in the zoo."""

    adapter_id: str
    family: str
    dataset: str
    metric: str
    rank: int
    lora_alpha: int
    target_modules: list[str]
    lora_dropout: float = 0.05
    # Measured offline in milestone 5: how much this adapter's score drops
    # per unit of rank truncation. Used by sensitivity_aware eviction.
    # None means "not yet measured", and the policy falls back to rank only.
    truncation_sensitivity: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def num_params(self, hidden_size: int, num_layers: int) -> int:
        """LoRA parameter count: two matrices (A and B) of shape r x hidden
        per targeted module per layer."""
        per_module = 2 * self.rank * hidden_size
        return per_module * len(self.target_modules) * num_layers

    def size_bytes(
        self, hidden_size: int, num_layers: int, dtype: str = "bfloat16"
    ) -> float:
        if dtype not in BYTES_PER_PARAM:
            raise ValueError(
                f"unknown adapter dtype {dtype!r}; "
                f"expected one of {sorted(BYTES_PER_PARAM)}"
            )
        return self.num_params(hidden_size, num_layers) * BYTES_PER_PARAM[dtype]


def build_zoo(config: dict[str, Any]) -> list[AdapterSpec]:
    """Construct the adapter zoo from the config.

    Adapters are split evenly across the task families, and ranks are drawn
    round-robin from `zoo.ranks` so the mixture is reproducible rather than
    random.
    """
    zoo_cfg = config["zoo"]
    families = zoo_cfg["task_families"]
    num_adapters = int(zoo_cfg["num_adapters"])
    if num_adapters < 1:
        raise ValueError("zoo.num_adapters must be at least 1")

    ranks = list(zoo_cfg["ranks"])
    homogeneous = zoo_cfg["rank_mixture"] == "homogeneous"
    alpha_ratio = float(zoo_cfg["lora_alpha_ratio"])

    specs: list[AdapterSpec] = []
    for i in range(num_adapters):
        family = families[i % len(families)]
        datasets = family["datasets"]
        dataset = datasets[(i // len(families)) % len(datasets)]

        rank = int(zoo_cfg["homogeneous_rank"]) if homogeneous else ranks[i % len(ranks)]

        specs.append(
            AdapterSpec(
                adapter_id=f"a{i:02d}_{family['name']}_r{rank}",
                family=family["name"],
                dataset=dataset,
                metric=family["metric"],
                rank=rank,
                lora_alpha=int(alpha_ratio * rank),
                target_modules=list(zoo_cfg["target_modules"]),
                lora_dropout=float(zoo_cfg["lora_dropout"]),
            )
        )
    return specs


def working_set_bytes(
    specs: list[AdapterSpec],
    hidden_size: int,
    num_layers: int,
    dtype: str = "bfloat16",
) -> float:
    """Total bytes of an adapter set. The memory budget B is expressed as a
    fraction of this, so that results transfer across GPUs."""
    return sum(s.size_bytes(hidden_size, num_layers, dtype) for s in specs)
