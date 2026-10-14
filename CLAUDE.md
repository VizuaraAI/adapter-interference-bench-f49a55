# CLAUDE.md

Instructions for the coding agent working on this repository with Venkata.

## Project goal

Measure and mitigate **adapter interference**: the gap between how a single
LoRA adapter behaves when it owns a GPU and how it behaves when it is
co-served with other adapters against one shared small base model under a
fixed memory budget.

The deliverable is a research paper (primary target: EuroMLSys 2027, with
arXiv as the floor) plus a reproducible benchmark. Treat the paper's results
table as the definition of done: every experiment should fill a cell in it.

## The one distinction to never lose

Three nearby topics are **not** this project. If a change starts drifting
toward them, say so:

- **Adapter routing** (which adapter should serve a query). We hold adapter
  assignment fixed and known. Routing is a different project.
- **Offline adapter merging** (permanently combining adapter weights). We
  study interference that appears during *serving*, plus the merging-style
  interference a scheduler transiently *chooses* under memory pressure.
- **Generic serving throughput optimisation**. Aggregate throughput is a cost
  we report, not the objective. The objective is per-adapter behaviour.

## Core definitions, keep these exact

- `IDS_i(S, B) = 1 - score_i(S, B) / score_i_solo`. Interference Degradation
  Score for adapter `i` co-served with set `S` under adapter-cache budget
  `B`. Higher is worse. Always report mean **and worst-adapter**.
- Tail slowdown: `p99_i(S, B) / p99_i_solo`, for both TTFT and TPOT.
- `B` is the adapter cache size as a fraction of the active working set's
  total adapter bytes, not an absolute byte count. Keep it relative so
  results transfer across GPUs.
- Resource channel versus semantic channel. Every reported degradation must
  be attributed to one of them. An unattributed number is not a result.

## Layout and conventions

```
src/adapters.py       zoo definition: task families, rank assignment
src/data.py           dataset loaders + per-task metric functions
src/train_adapter.py  train one adapter on the shared frozen base
src/serving.py        ServingBackend protocol; real + simulator impls
src/policies.py       eviction policies, interference-aware scheduler
src/metrics.py        IDS, tail slowdown, fairness, answer drift
src/measure.py        sweep driver, CLI entry point
configs/default.yaml  every experiment knob lives here
```

Conventions:

- **All experiment knobs go in `configs/default.yaml`.** No magic numbers in
  `src/`. If you need a new knob, add it to the config with a comment and a
  sensible default.
- Every function that computes a metric takes explicit arguments and returns
  a plain dict or dataclass. No hidden global state, because these numbers go
  in a paper and need to be auditable.
- `ServingBackend` is a protocol with two implementations: `VLLMBackend`
  (real measurements) and `SimulatorBackend` (analytic cache and queue
  model). New policies must work against both. The simulator exists so the
  student can iterate without burning GPU hours; keep it honest about what it
  does and does not model (it has no quality channel at all).
- Results are append-only CSV in `results/`. Never overwrite a results file;
  add a run id. Lost measurements cost GPU hours.
- Determinism: fix seeds, and when measuring answer drift, that is the one
  place where non-determinism is the signal rather than noise. Do not
  "fix" it there.

## Writing style for any text you produce

No em dashes. Use commas, colons, parentheses or semicolons. This applies to
code comments, docstrings, README edits, and especially paper text.

## Next steps, in order

1. **Make the simulator path green.** `python -m src.measure --simulate
   --adapters 4` should produce a complete metric table. This is the
   scaffolding check.
2. **Train 4 real adapters** on 2 task families with rank 8, and reproduce
   the solo reference numbers. Verify that solo IDS is exactly zero by
   construction.
3. **The critical first experiment**: co-serve those 4 adapters at
   `B = 100%` with unmerged adapter application. Confirm mean IDS is
   approximately zero while tail slowdown is clearly above 1. This validates
   hypothesis H1 and, just as importantly, validates the harness. Non-zero
   IDS here means a bug.
4. **Scale the zoo to 32 adapters with mixed ranks** `{4, 8, 16, 32}`, then
   sweep `B` over `{100%, 50%, 25%, 12%}`. This produces the
   memory-interference frontier, which is figure 2 of the paper.
5. **Implement the fallback ladder** (quantisation, rank truncation, grouped
   merging) with one mechanism enabled at a time so each one's accuracy cost
   is separately attributable. This is the semantic channel.
6. **Implement the interference-aware policy**: adapter concurrency cap at
   admission, sensitivity-aware eviction, explicit pool split. Then the
   concurrency-cap sweep to find where throughput loss overtakes
   worst-adapter gain.
7. **Replicate the headline configuration** on SmolLM2-1.7B and
   Llama-3.2-1B. Only claim findings that hold on all three bases.

## When asked to add an experiment

Ask which cell of the paper's results table it fills. If it does not fill
one, it is probably a distraction from an eight-week timeline. Prefer
deepening an existing axis over adding a new one.
