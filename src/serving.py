"""Serving backends.

Two implementations behind one protocol:

- `SimulatorBackend`: an analytic cache and queue model. Runs on CPU with
  only the core requirements installed. It models the RESOURCE channel only.
  It has no quality channel at all, so IDS is zero by construction. That is
  a feature: it lets you debug the harness without burning GPU hours, and it
  gives you a known-answer test (if the simulator reports non-zero IDS, the
  harness is broken).
- `VLLMBackend`: real measurements against vLLM's multi-LoRA support. This is
  where every number in the paper comes from.

New policies must work against both.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Protocol

from .adapters import AdapterSpec
from .metrics import AdapterObservation
from .policies import AdapterCache, InterferenceAwareScheduler


@dataclass
class Request:
    request_id: int
    adapter_id: str
    prompt: str
    target: str
    prompt_tokens: int
    output_tokens: int


@dataclass
class FallbackConfig:
    """Which budget-saving mechanism is active. Exactly one at a time, so its
    accuracy cost is separately attributable to it."""

    kind: str = "none"
    truncate_rank_to: int = 4
    grouped_merge_k: int = 4

    VALID = (
        "none",
        "quantise_8bit",
        "quantise_4bit",
        "truncate_rank",
        "grouped_merge",
    )

    def __post_init__(self) -> None:
        if self.kind not in self.VALID:
            raise ValueError(
                f"unknown fallback {self.kind!r}; expected one of {self.VALID}"
            )

    def adapter_dtype(self) -> str:
        if self.kind == "quantise_8bit":
            return "int8"
        if self.kind == "quantise_4bit":
            return "int4"
        return "bfloat16"

    def effective_rank(self, rank: int) -> int:
        if self.kind == "truncate_rank":
            return min(rank, self.truncate_rank_to)
        return rank


class ServingBackend(Protocol):
    name: str

    def run(
        self,
        specs: list[AdapterSpec],
        requests: list[Request],
        cache: AdapterCache,
        scheduler: InterferenceAwareScheduler,
        fallback: FallbackConfig,
    ) -> dict[str, AdapterObservation]: ...


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class SimulatorBackend:
    """Analytic model of the resource channel.

    Latency for a request is the sum of:
      - queue wait, driven by how many requests the concurrency cap defers,
      - adapter load time on a cache miss, proportional to adapter bytes,
      - base model forward time, discounted by batch efficiency.

    Quality is passed through unchanged from the solo score, which is why
    IDS is exactly zero here.
    """

    name = "simulator"

    def __init__(self, config: dict[str, Any], solo_scores: dict[str, float]):
        sim = config["simulator"]
        self.load_ms_per_mb = float(sim["adapter_load_ms_per_mb"])
        self.forward_ms_per_token = float(sim["base_forward_ms_per_token"])
        self.batch_efficiency = float(sim["batch_efficiency"])
        self.solo_scores = solo_scores
        self.hidden_size = 1536
        self.num_layers = 28

    def run(
        self,
        specs: list[AdapterSpec],
        requests: list[Request],
        cache: AdapterCache,
        scheduler: InterferenceAwareScheduler,
        fallback: FallbackConfig,
    ) -> dict[str, AdapterObservation]:
        by_id = {s.adapter_id: s for s in specs}
        observations: dict[str, AdapterObservation] = {
            s.adapter_id: AdapterObservation(
                adapter_id=s.adapter_id,
                rank=fallback.effective_rank(s.rank),
                task_score=self.solo_scores.get(s.adapter_id, 0.0),
                ttft_s=[],
                tpot_ms=[],
                num_requests=0,
            )
            for s in specs
        }

        pending = list(requests)
        clock_s = 0.0

        while pending:
            adapter_ids = [r.adapter_id for r in pending]
            admitted_idx, deferred = scheduler.form_batch(adapter_ids)
            if not admitted_idx:
                raise RuntimeError(
                    "scheduler admitted nothing while requests remain; check "
                    "policy.adapter_concurrency_cap"
                )

            batch = [pending[i] for i in admitted_idx]
            distinct = {r.adapter_id for r in batch}

            # Adapter cache cost for this batch.
            load_s = 0.0
            for adapter_id in sorted(distinct):
                spec = by_id[adapter_id]
                size_bytes = spec.size_bytes(
                    self.hidden_size, self.num_layers, fallback.adapter_dtype()
                )
                hit = cache.access(spec, size_bytes)
                observation = observations[adapter_id]
                if hit:
                    observation.cache_hits += 1
                else:
                    observation.cache_misses += 1
                    observation.swap_bytes += size_bytes
                    load_s += (size_bytes / 1e6) * self.load_ms_per_mb / 1000.0

            # Base model cost, shared across the batch.
            prompt_tokens = sum(r.prompt_tokens for r in batch)
            prefill_s = (
                prompt_tokens * self.forward_ms_per_token * self.batch_efficiency
            ) / 1000.0

            for request in batch:
                observation = observations[request.adapter_id]
                observation.ttft_s.append(clock_s + load_s + prefill_s)
                per_token_ms = (
                    self.forward_ms_per_token
                    * len(batch)
                    * self.batch_efficiency
                )
                observation.tpot_ms.append(per_token_ms)
                observation.num_requests += 1

            clock_s += load_s + prefill_s
            # Decode phase advances the clock for the longest request.
            longest = max(r.output_tokens for r in batch)
            clock_s += (
                longest * self.forward_ms_per_token * len(batch)
                * self.batch_efficiency
            ) / 1000.0

            remaining = {i for i in range(len(pending))} - set(admitted_idx)
            pending = [pending[i] for i in sorted(remaining)]

        return observations


# ---------------------------------------------------------------------------
# Real backend
# ---------------------------------------------------------------------------


class VLLMBackend:
    """Real multi-LoRA serving measurements.

    Every number reported in the paper comes from this backend. It keeps one
    copy of the frozen base model and applies adapters per request without
    merging them into the base, which is the configuration under which
    hypothesis H1 predicts zero quality interference.
    """

    name = "vllm"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.base_name = config["base_model"]["name"]
        self.max_model_len = int(config["base_model"]["max_model_len"])
        self._engine = None

    def _ensure_engine(self, max_rank: int, max_loras: int) -> Any:
        if self._engine is not None:
            return self._engine
        from vllm import LLM  # lazy: only needed for real runs

        self._engine = LLM(
            model=self.base_name,
            enable_lora=True,
            max_lora_rank=max_rank,
            max_loras=max_loras,
            max_model_len=self.max_model_len,
            dtype=self.config["base_model"]["dtype"],
        )
        return self._engine

    def run(
        self,
        specs: list[AdapterSpec],
        requests: list[Request],
        cache: AdapterCache,
        scheduler: InterferenceAwareScheduler,
        fallback: FallbackConfig,
    ) -> dict[str, AdapterObservation]:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        from .data import score

        artifacts = self.config["run"]["artifacts_dir"]
        by_id = {s.adapter_id: s for s in specs}
        max_rank = max(fallback.effective_rank(s.rank) for s in specs)

        engine = self._ensure_engine(
            max_rank=max_rank, max_loras=scheduler.adapter_concurrency_cap
        )
        sampling = SamplingParams(temperature=0.0, max_tokens=256)

        lora_requests = {
            spec.adapter_id: LoRARequest(
                lora_name=spec.adapter_id,
                lora_int_id=index + 1,
                lora_path=f"{artifacts}/adapters/{spec.adapter_id}",
            )
            for index, spec in enumerate(specs)
        }

        observations: dict[str, AdapterObservation] = {
            s.adapter_id: AdapterObservation(
                adapter_id=s.adapter_id,
                rank=fallback.effective_rank(s.rank),
                task_score=0.0,
                ttft_s=[],
                tpot_ms=[],
                num_requests=0,
            )
            for s in specs
        }
        predictions: dict[str, list[str]] = {s.adapter_id: [] for s in specs}
        targets: dict[str, list[str]] = {s.adapter_id: [] for s in specs}

        pending = list(requests)
        while pending:
            admitted_idx, _ = scheduler.form_batch([r.adapter_id for r in pending])
            batch = [pending[i] for i in admitted_idx]

            # Account the adapter cache for this batch's distinct adapters so
            # that hit rate and swap bytes are comparable with the simulator.
            for adapter_id in sorted({r.adapter_id for r in batch}):
                spec = by_id[adapter_id]
                size_bytes = spec.size_bytes(
                    _hidden_size(engine), _num_layers(engine), fallback.adapter_dtype()
                )
                hit = cache.access(spec, size_bytes)
                if hit:
                    observations[adapter_id].cache_hits += 1
                else:
                    observations[adapter_id].cache_misses += 1
                    observations[adapter_id].swap_bytes += size_bytes

            # vLLM batches internally; we group by adapter because one
            # generate() call takes a single LoRARequest.
            for adapter_id in sorted({r.adapter_id for r in batch}):
                group = [r for r in batch if r.adapter_id == adapter_id]
                start = time.perf_counter()
                outputs = engine.generate(
                    [r.prompt for r in group],
                    sampling,
                    lora_request=lora_requests[adapter_id],
                )
                elapsed = time.perf_counter() - start

                observation = observations[adapter_id]
                for request, output in zip(group, outputs):
                    text = output.outputs[0].text
                    generated = max(1, len(output.outputs[0].token_ids))
                    predictions[adapter_id].append(text)
                    targets[adapter_id].append(request.target)
                    # Wall clock split across the group; replace with vLLM's
                    # per-request metrics once you move to the async engine.
                    observation.ttft_s.append(elapsed / len(group))
                    observation.tpot_ms.append(1000.0 * elapsed / generated)
                    observation.num_requests += 1

            remaining = {i for i in range(len(pending))} - set(admitted_idx)
            pending = [pending[i] for i in sorted(remaining)]

        for spec in specs:
            if predictions[spec.adapter_id]:
                observations[spec.adapter_id].task_score = score(
                    spec.metric,
                    predictions[spec.adapter_id],
                    targets[spec.adapter_id],
                )
        return observations


def _hidden_size(engine: Any) -> int:
    try:
        return int(engine.llm_engine.model_config.hf_config.hidden_size)
    except AttributeError:
        return 1536


def _num_layers(engine: Any) -> int:
    try:
        return int(engine.llm_engine.model_config.hf_config.num_hidden_layers)
    except AttributeError:
        return 28


def build_workload(
    specs: list[AdapterSpec],
    config: dict[str, Any],
    skew: str,
    rng: random.Random,
) -> list[Request]:
    """Generate the request stream.

    Skew matters because it is what makes opportunistic merging attractive,
    and therefore what triggers the semantic interference channel.
    """
    workload = config["workload"]
    num_requests = int(workload["num_requests"])

    if skew == "uniform":
        weights = [1.0] * len(specs)
    elif skew == "zipf":
        alpha = float(config["sweep"]["zipf_alpha"])
        weights = [1.0 / ((i + 1) ** alpha) for i in range(len(specs))]
    else:
        raise ValueError(f"unknown load skew {skew!r}; expected uniform or zipf")

    requests: list[Request] = []
    for request_id in range(num_requests):
        spec = rng.choices(specs, weights=weights, k=1)[0]
        requests.append(
            Request(
                request_id=request_id,
                adapter_id=spec.adapter_id,
                prompt=f"[{spec.dataset}] placeholder prompt {request_id}",
                target="",
                prompt_tokens=int(workload["prompt_tokens_mean"]),
                output_tokens=int(workload["output_tokens_mean"]),
            )
        )
    return requests
