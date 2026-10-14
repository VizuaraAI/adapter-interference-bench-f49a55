# Experiment log

One row per run. The columns are deliberately the same quantities as the
main results table in the paper draft (`out/paper/main.tex`,
Table~\ref{tab:main}), so that filling this log fills the paper.

Copy the blank row, do not edit old rows. If a run was wrong, add a new row
and write why in the notes. A log you trust is worth more than a tidy one.

## How to fill this in

- **Config hash**: `git rev-parse --short HEAD` plus any config override you
  passed. Every row must be reproducible from these two things alone.
- **Channel**: which interference channel this run is probing, `resource`
  or `semantic`. A run that cannot be assigned to one is not an experiment
  yet, it is an observation.
- **Mean / worst IDS**: from `results/table.md`. Always record both. The
  worst-adapter number is the one that goes in the abstract.
- **Verdict**: what this run actually decided. "Looks interesting" is not a
  verdict. "H1 holds at B=100%, mean IDS 0.00 within noise" is.

## Milestone 1: harness validation

| Date | Config hash | Backend | N | B | Skew | Fallback | Eviction | Cap | Channel | Mean IDS | Worst IDS | p99 TTFT (s) | Worst TTFT slowdown | SLO % | Verdict / notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  | simulator | 4 | 1.0 | uniform | none | lru | 4 | resource | 0.0000 | 0.0000 |  |  |  | Scaffolding check. IDS must be exactly 0 by construction; anything else is a bug. |
|  |  | vllm | 1 | 1.0 | uniform | none | n/a | 1 | n/a | 0.0000 | 0.0000 |  | 1.00x |  | Solo reference. Record the raw per-adapter scores here, they anchor every later row. |

## Milestone 2: H1, resource interference at a generous budget

Prediction: mean IDS indistinguishable from zero, tail slowdown clearly
above 1. This is simultaneously the first result and the harness
correctness check.

| Date | Config hash | Backend | N | B | Skew | Fallback | Eviction | Cap | Channel | Mean IDS | Worst IDS | p99 TTFT (s) | Worst TTFT slowdown | SLO % | Verdict / notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  | vllm | 4 | 1.0 | uniform | none | lru | 4 | resource |  |  |  |  |  |  |
|  |  | vllm | 16 | 1.0 | uniform | none | lru | 16 | resource |  |  |  |  |  |  |
|  |  | vllm | 16 | 1.0 | zipf | none | lru | 16 | resource |  |  |  |  |  |  |

## Milestone 3: the memory-interference frontier

Prediction: the worst-adapter curve separates from the mean curve before the
mean becomes alarming. This table becomes figure 2.

| Date | Config hash | Backend | N | B | Skew | Fallback | Eviction | Cap | Channel | Mean IDS | Worst IDS | p99 TTFT (s) | Worst TTFT slowdown | SLO % | Verdict / notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  | vllm | 16 | 1.0 | zipf | none | lru | 16 | both |  |  |  |  |  |  |
|  |  | vllm | 16 | 0.5 | zipf | none | lru | 16 | both |  |  |  |  |  |  |
|  |  | vllm | 16 | 0.25 | zipf | none | lru | 16 | both |  |  |  |  |  |  |
|  |  | vllm | 16 | 0.12 | zipf | none | lru | 16 | both |  |  |  |  |  |  |

## Milestone 4: mechanism attribution (the semantic channel)

One fallback enabled at a time, so each one's accuracy cost is separately
attributable. Deliverable: a ranked list of which budget-saving mechanism a
practitioner should reach for first.

| Date | Config hash | Backend | N | B | Skew | Fallback | Eviction | Cap | Channel | Mean IDS | Worst IDS | p99 TTFT (s) | Worst TTFT slowdown | SLO % | Verdict / notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  | vllm | 16 | 0.25 | zipf | none | lru | 16 | semantic |  |  |  |  |  | Control. |
|  |  | vllm | 16 | 0.25 | zipf | quantise_8bit | lru | 16 | semantic |  |  |  |  |  |  |
|  |  | vllm | 16 | 0.25 | zipf | quantise_4bit | lru | 16 | semantic |  |  |  |  |  |  |
|  |  | vllm | 16 | 0.25 | zipf | truncate_rank | lru | 16 | semantic |  |  |  |  |  |  |
|  |  | vllm | 16 | 0.25 | zipf | grouped_merge | lru | 16 | semantic |  |  |  |  |  | Expected to be the largest single source. |

### H3: answer drift from batch composition alone

Same adapter, same prompts, same budget, different batch composition. This
isolates numerics from policy. Record the drift percentage, not IDS.

| Date | Config hash | N | Batch composition A | Batch composition B | Answer drift % | Notes |
|---|---|---|---|---|---|---|
|  |  | 16 | cap=16, mixed | cap=1, isolated | | Bounds how much observed quality change can be blamed on numerics. |

## Milestone 5: eviction policy comparison

Separates "evicts less" from "evicts better".

| Date | Config hash | Backend | N | B | Skew | Fallback | Eviction | Cap | Channel | Mean IDS | Worst IDS | Hit rate % | Jain's index | SLO % | Verdict / notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  | vllm | 16 | 0.25 | zipf | none | lru | 16 | both |  |  |  |  |  |  |
|  |  | vllm | 16 | 0.25 | zipf | none | lfu | 16 | both |  |  |  |  |  |  |
|  |  | vllm | 16 | 0.25 | zipf | none | rank_aware | 16 | both |  |  |  |  |  | Expect best hit rate, worst high-rank tenant. |
|  |  | vllm | 16 | 0.25 | zipf | none | sensitivity_aware | 16 | both |  |  |  |  |  |  |

## Milestone 6: the proposed policy and its cost

Prediction: a broad flat region in the cap sweep, so the policy is easy to
tune. Record throughput here, because the honest claim is "most of the
worst-case damage is recoverable, and here is what it costs".

| Date | Config hash | Backend | N | B | Skew | Eviction | Cap | Worst IDS | Worst TTFT slowdown | Throughput (req/s) | Throughput vs unconstrained | Verdict / notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  | vllm | 16 | 0.25 | zipf | sensitivity_aware | 1 |  |  |  |  | Full isolation, the upper bound on quality stability. |
|  |  | vllm | 16 | 0.25 | zipf | sensitivity_aware | 2 |  |  |  |  |  |
|  |  | vllm | 16 | 0.25 | zipf | sensitivity_aware | 4 |  |  |  |  |  |
|  |  | vllm | 16 | 0.25 | zipf | sensitivity_aware | 8 |  |  |  |  |  |
|  |  | vllm | 16 | 0.25 | zipf | sensitivity_aware | 16 |  |  |  |  | Unconstrained, what existing servers do. |

## Milestone 7: base model transfer

Only claim findings that hold on all three bases.

| Date | Config hash | Base model | N | B | Skew | Eviction | Cap | Mean IDS | Worst IDS | Worst TTFT slowdown | Holds? | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  | Qwen2.5-1.5B-Instruct | 16 | 0.25 | zipf | sensitivity_aware | 4 |  |  |  |  | Primary. |
|  |  | SmolLM2-1.7B-Instruct | 16 | 0.25 | zipf | sensitivity_aware | 4 |  |  |  |  |  |
|  |  | Llama-3.2-1B-Instruct | 16 | 0.25 | zipf | sensitivity_aware | 4 |  |  |  |  |  |

## Blank row to copy

| Date | Config hash | Backend | N | B | Skew | Fallback | Eviction | Cap | Channel | Mean IDS | Worst IDS | p99 TTFT (s) | Worst TTFT slowdown | SLO % | Verdict / notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Running list of surprises

Anything that did not match the prediction goes here immediately, with the
date. Surprises are where the paper's actual contribution usually comes
from, and they are also the first thing you forget.

| Date | What I expected | What happened | What I did about it |
|---|---|---|---|
|  |  |  |  |
