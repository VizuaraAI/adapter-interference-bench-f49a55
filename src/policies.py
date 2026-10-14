"""Eviction policies and the interference-aware scheduler.

The adapter cache is a fixed-byte pool holding LoRA weights of mixed rank.
A policy decides, on a miss, which resident adapter to evict. The point of
this module is that "evict less" and "evict better" are different goals: LRU
minimises misses, while sensitivity-aware eviction minimises the damage each
miss does to the tenant who owns the evicted adapter.

Every policy here must work against both serving backends in src/serving.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .adapters import AdapterSpec


@dataclass
class CacheEntry:
    spec: AdapterSpec
    size_bytes: float
    last_used_step: int
    use_count: int = 0


class EvictionPolicy(Protocol):
    """Returns the adapter_id that should be evicted."""

    name: str

    def choose_victim(
        self, entries: dict[str, CacheEntry], step: int, weights: dict[str, float]
    ) -> str: ...


class LRUEviction:
    name = "lru"

    def choose_victim(
        self, entries: dict[str, CacheEntry], step: int, weights: dict[str, float]
    ) -> str:
        return min(entries, key=lambda k: entries[k].last_used_step)


class LFUEviction:
    name = "lfu"

    def choose_victim(
        self, entries: dict[str, CacheEntry], step: int, weights: dict[str, float]
    ) -> str:
        return min(entries, key=lambda k: entries[k].use_count)


class RankAwareEviction:
    """Evict the largest adapter first, which maximises bytes freed per
    eviction. This is the policy that looks best on aggregate throughput and
    is expected to be worst for high-rank tenants."""

    name = "rank_aware"

    def choose_victim(
        self, entries: dict[str, CacheEntry], step: int, weights: dict[str, float]
    ) -> str:
        return max(entries, key=lambda k: entries[k].size_bytes)


class SensitivityAwareEviction:
    """Evict the adapter whose eviction is cheapest, where cost combines
    recency, size, and the adapter's measured truncation sensitivity.

    Adapters with unmeasured sensitivity get the neutral value 1.0, so the
    policy degrades gracefully to a recency-and-rank policy before the
    offline sensitivity measurement of milestone 5 has been done.
    """

    name = "sensitivity_aware"

    def choose_victim(
        self, entries: dict[str, CacheEntry], step: int, weights: dict[str, float]
    ) -> str:
        def cost(key: str) -> float:
            entry = entries[key]
            staleness = float(step - entry.last_used_step)
            recency_term = weights.get("recency", 1.0) / (1.0 + staleness)
            rank_term = weights.get("rank", 0.5) * (entry.spec.rank / 32.0)
            sensitivity = entry.spec.truncation_sensitivity
            sensitivity_term = weights.get("sensitivity", 1.5) * (
                1.0 if sensitivity is None else float(sensitivity)
            )
            return recency_term + rank_term + sensitivity_term

        return min(entries, key=cost)


EVICTION_POLICIES: dict[str, type] = {
    LRUEviction.name: LRUEviction,
    LFUEviction.name: LFUEviction,
    RankAwareEviction.name: RankAwareEviction,
    SensitivityAwareEviction.name: SensitivityAwareEviction,
}


def make_eviction_policy(name: str) -> EvictionPolicy:
    if name not in EVICTION_POLICIES:
        raise ValueError(
            f"unknown eviction policy {name!r}; "
            f"expected one of {sorted(EVICTION_POLICIES)}"
        )
    return EVICTION_POLICIES[name]()


@dataclass
class AdapterCache:
    """A fixed-byte adapter cache with a pluggable eviction policy.

    Tracks hits, misses and swapped bytes, which together are the resource
    interference channel.
    """

    capacity_bytes: float
    policy: EvictionPolicy
    eviction_weights: dict[str, float] = field(default_factory=dict)
    entries: dict[str, CacheEntry] = field(default_factory=dict)
    step: int = 0
    hits: int = 0
    misses: int = 0
    swap_bytes: float = 0.0
    evictions: int = 0

    def access(self, spec: AdapterSpec, size_bytes: float) -> bool:
        """Touch an adapter. Returns True on a cache hit.

        On a miss the adapter is loaded, evicting victims chosen by the
        policy until it fits.
        """
        self.step += 1
        key = spec.adapter_id

        if key in self.entries:
            entry = self.entries[key]
            entry.last_used_step = self.step
            entry.use_count += 1
            self.hits += 1
            return True

        self.misses += 1
        if size_bytes > self.capacity_bytes:
            raise ValueError(
                f"adapter {key!r} needs {size_bytes:.0f} bytes but the cache "
                f"capacity is {self.capacity_bytes:.0f}; raise the budget or "
                "lower the adapter rank"
            )

        while self._resident_bytes() + size_bytes > self.capacity_bytes:
            victim = self.policy.choose_victim(
                self.entries, self.step, self.eviction_weights
            )
            del self.entries[victim]
            self.evictions += 1

        self.entries[key] = CacheEntry(
            spec=spec, size_bytes=size_bytes, last_used_step=self.step, use_count=1
        )
        self.swap_bytes += size_bytes
        return False

    def _resident_bytes(self) -> float:
        return sum(e.size_bytes for e in self.entries.values())

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": (100.0 * self.hits / total) if total else float("nan"),
            "evictions": self.evictions,
            "swap_bytes": self.swap_bytes,
            "resident_bytes": self._resident_bytes(),
        }


@dataclass
class InterferenceAwareScheduler:
    """Admission control that caps how many distinct adapters may appear in
    one decode batch.

    cap = 1 is full isolation (no mixed batches, maximum quality stability,
    worst throughput). cap = N is unconstrained, which is what existing
    multi-LoRA servers do. The cap sweep in milestone 6 looks for the broad
    flat region in between.
    """

    adapter_concurrency_cap: int
    max_batch_requests: int = 32

    def form_batch(self, pending: list[str]) -> tuple[list[int], list[str]]:
        """Pick request indices for the next batch from a pending queue of
        adapter ids, preserving arrival order within the admitted set.

        Returns (admitted_indices, deferred_adapter_ids).
        """
        if self.adapter_concurrency_cap < 1:
            raise ValueError("policy.adapter_concurrency_cap must be at least 1")

        admitted: list[int] = []
        admitted_adapters: list[str] = []
        deferred: list[str] = []

        for index, adapter_id in enumerate(pending):
            if len(admitted) >= self.max_batch_requests:
                deferred.append(adapter_id)
                continue
            if adapter_id in admitted_adapters:
                admitted.append(index)
                continue
            if len(admitted_adapters) < self.adapter_concurrency_cap:
                admitted_adapters.append(adapter_id)
                admitted.append(index)
            else:
                deferred.append(adapter_id)

        return admitted, deferred


def split_pool(
    pool_bytes: float, adapter_floor: float, kv_floor: float
) -> tuple[float, float]:
    """Divide the unified paged pool between the adapter cache and the KV
    cache using explicit reserved floors.

    Unified paging deliberately lets these two consumers compete freely,
    which is the mechanism behind the resource channel. This function is the
    knob that stops them competing.
    """
    if adapter_floor < 0 or kv_floor < 0:
        raise ValueError("pool floors must be non-negative")
    if adapter_floor + kv_floor > 1.0:
        raise ValueError(
            f"pool floors sum to {adapter_floor + kv_floor:.2f}, which exceeds "
            "the pool; lower policy.adapter_cache_floor or policy.kv_cache_floor"
        )
    slack = 1.0 - adapter_floor - kv_floor
    # Give the slack to the adapter cache, since KV pressure is already
    # handled by the backend's own preemption path.
    return pool_bytes * (adapter_floor + slack), pool_bytes * kv_floor
