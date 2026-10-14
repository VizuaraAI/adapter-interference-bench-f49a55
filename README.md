# adapter-interference-bench

Measuring what one tenant actually pays when many LoRA adapters share one
small base model under a memory budget.

This is the starter repository for Venkata Narayana's Vizuara SLM Bootcamp
research project, **Adapter Interference on a Shared Small Base**.

## The question this repo exists to answer

Multi-LoRA serving systems (S-LoRA, Punica, dLoRA, vLLM's multi-LoRA support)
have shown that one base model can serve a great many adapters. They report
aggregate throughput and average latency. They do not report what happens to
*one specific adapter* when it is forced to share.

So we measure the counterfactual directly:

```
IDS_i(S, B) = 1 - score_i(co-served with S, budget B) / score_i(solo)
```

An adapter is served alone, then co-served with `N-1` neighbours under an
adapter-cache budget `B`. The difference is interference. We split it into
two channels:

- **Resource interference**: queueing, adapter swap traffic, and contention
  between the adapter cache and the KV cache for the same paged pool. Shows
  up as tail latency and SLO misses.
- **Semantic interference**: the answer itself changes. This should only
  happen once the budget forces a fallback (rank truncation, adapter
  quantisation, grouped merging), plus a small floor from
  batch-composition-dependent numerics.

Note what this is *not*. It is not adapter routing (which adapter should
answer). It is not offline adapter merging (permanently combining weights).
The adapter assignment is fixed and known; we vary only co-tenancy and the
memory budget.

## Setup

Requires Python 3.10+ and one CUDA GPU (24 GB is the reference platform,
16 GB works for the reduced sweep). CPU-only works for the simulator and the
unit tests, but not for real serving measurements.

```bash
git clone <your-fork-url> adapter-interference-bench
cd adapter-interference-bench

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Nothing large is committed here. The dataset loaders pull public datasets
from the Hugging Face Hub on first use and cache them under
`~/.cache/huggingface`.

## The first three things to run

### 1. Confirm the harness works end to end (about 2 minutes, no GPU needed)

```bash
python -m src.measure --config configs/default.yaml --simulate --adapters 4
```

The `--simulate` flag replaces the real serving backend with an analytic
cache/queue model. It produces the same metric table shape as a real run, so
you can debug the plumbing before you spend GPU hours. You should see one row
per configuration with `mean_ids` and `worst_ids` at exactly 0.0000 (the
simulator has no quality channel) and tail slowdowns at or just above 1.00x.
The zeros are the point: they are your known-answer test. If this path ever
reports non-zero IDS, you have a bug in the harness, not a finding.

There is also a fuller check that runs the imports, prints the zoo, and then
runs the sweep:

```bash
bash scripts/smoke_test.sh
```

### 2. Train a small adapter zoo (about 1 to 2 hours on one GPU)

```bash
bash scripts/train_zoo.sh
```

Trains the adapter zoo defined in `configs/default.yaml`. Start with
`num_adapters: 4` and `max_train_samples: 500` to get something working
today, then scale to 32 adapters with mixed ranks once the pipeline is
trustworthy. Adapters land in `artifacts/adapters/<adapter_id>/`.

### 3. Run the real sweep and get your first interference number

```bash
bash scripts/run_sweep.sh
```

This runs the solo reference for every adapter, then every co-served
configuration in the sweep, and writes `results/sweep.csv` plus a rendered
table to `results/table.md`. The columns of that table map one-to-one onto
the main results table in the paper draft, and onto the rows of
`experiments/log_template.md`.

## Layout

```
src/
  adapters.py     adapter zoo definition, task families, rank assignment
  data.py         dataset loading and per-task metric functions
  train_adapter.py  train one LoRA adapter on the shared frozen base
  serving.py      serving backends: real (vLLM) and analytic simulator
  policies.py     eviction policies and the interference-aware scheduler
  metrics.py      IDS, tail slowdown, fairness, answer drift
  measure.py      the sweep driver and entry point
configs/
  default.yaml    the single source of truth for every experiment knob
scripts/
  train_zoo.sh    train all adapters in the zoo
  run_sweep.sh    run the full solo-plus-co-served sweep
experiments/
  log_template.md one row per experiment, mapped to the paper's table
```

## How to read the results

The number that matters is **worst-adapter IDS**, not mean IDS. A policy can
look excellent on the mean while destroying one tenant. If you only ever
report means, you have reproduced the gap in the literature rather than
closing it.

Expect this shape: at `B = 100%` the IDS should be approximately zero while
tail slowdown is already significant. That is the headline finding, and it is
also your correctness check: if you see non-zero IDS at a generous budget
with unmerged adapters, you have a bug, not a result.

## Where this is going

Primary target venue is EuroMLSys 2027 (co-located with EuroSys 2027), with
arXiv as the floor. See the research kit's roadmap for the week-by-week plan
and which paper section each milestone fills in.
