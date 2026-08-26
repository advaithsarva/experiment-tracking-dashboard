# Experiment Tracking Dashboard

Log training runs to SQLite, compare them, and get told which ones are going
wrong — early enough that killing them saves something.

Python standard library only. No server, no account, no `pip install`, no
external service.

**Across 60 seeded runs with known outcomes: 100% of pathological runs
detected, 55.1% of their compute saved versus running to completion — and a
16.7% false-positive rate on healthy runs, which is the number worth arguing
about.** Full method in [RESULTS.md](RESULTS.md).

---

## The rule the whole thing turns on

> **A value that `log_metric` returned from is on disk. If the process dies
> one instruction later, that value survives.**

This is the entire reason an experiment tracker exists. You log precisely so
that when a 6-hour run dies at hour 5 you can see what happened before it
died. A tracker that buffers metrics in memory and flushes "periodically"
loses exactly the tail you need — the last steps before the crash, which are
the ones that explain it.

So the default is `durability="commit"`: every `log_metric` is its own
committed transaction. `test_metrics_survive_a_killed_process` spawns a child,
logs 50 metrics, and kills it with `os._exit(1)` — no `atexit`, no flush, no
`__exit__` — then asserts all 50 are readable.

That costs throughput, and the cost is measured rather than hand-waved:
**8,704 metrics/sec committed vs 64,232 batched, 7.4× slower.**
`durability="batch"` is available for tight inner loops and is honest about
what it gives up.

---

## Using it

```python
from tracker import Run

with Run("resnet-lr-sweep", params={"lr": 3e-4, "batch": 64, "layers": [64, 32]}) as run:
    for step in range(2000):
        run.log_metric("train_loss", loss, step)
        if step % 50 == 0:
            run.log_metric("val_loss", validate(), step)
    run.log_artifact("checkpoint.pt")
```

The context manager is not sugar. Without it a crashed run stays marked
`running` forever and the dashboard cannot tell "still going" from "died three
days ago" — which is the question people ask most. On an exception it records
`crashed` plus the error, then re-raises.

```bash
python cli.py runs                          # list
python cli.py show <run-id>                 # one run in full
python cli.py compare <id> <id> <id>        # side by side
python cli.py diagnose                      # health of every run
python cli.py dashboard board.html          # everything as one page
python cli.py export runs.csv
```

`--json` works on any subcommand, on either side of it, and writes one JSON
document to stdout and nothing else.

---

## The dashboard

One self-contained HTML file, no build step, no `node_modules`. Curves are
inline SVG polylines generated from the database.

Non-finite values **break the line rather than being dropped**. A gap is
honest about where the loss went NaN; skipping the point draws a smooth curve
straight through the most important event in the run.

---

## Diagnosis, and the number that makes it a claim

Anyone can tell a run failed by looking at the finished curve. The only useful
question is **how early**, because the point is to kill it and get the GPU
back. So `diagnose_incrementally()` replays the run step by step and reports
the first point at which the problem became detectable.

Every claim therefore has two numbers:

| | Meaning |
|---|---|
| detection rate | of runs that really were pathological, how many were caught |
| **compute saved** | of all training steps, how many killing it would avoid |

A detector that catches everything at 95% of the way through has a perfect
detection rate and saves nothing. The baseline — "wait until the end" — has
100% accuracy and saves 0%, by definition. Both numbers, always, or the first
one is marketing.

Four failure modes, each a rule over the curve:

| Kind | Signal | Suggestion it gives |
|---|---|---|
| `diverged` | NaN, infinity, or sustained growth above 3× best | lower LR, clip gradients |
| `overfitting` | val ≥2% above its best for 4 evals while train keeps gaining | stop, take the best checkpoint |
| `plateau` | half-to-half trend under 0.5% over 40 points | decay LR, or stop |
| `unstable` | mean step-to-step swing > 2× net descent | lower LR or raise batch size |

Ground truth comes from `simulate.py`, which decides each run's pathology
*before* generating its curve. Without that label, "detection rate" would be a
number with nothing behind it.

---

## The bug the tests caught

The detector started at **11/20** on labelled runs. Four different symptoms:

- `unstable` runs reported as `diverged`
- `plateau` runs reported as `healthy`
- an `overfitting` run missed entirely
- later, `healthy` runs reported as `overfitting`

**One root cause.** Every rule compared raw adjacent values, and real loss
curves carry multiplicative noise. So one lucky spike read as divergence, one
unlucky dip reset the overfitting counter forever, and noise in a flat curve
masked a plateau.

The fix was a single shared helper — a moving **median** (not mean: one 1e12
spike destroys a mean and leaves a median untouched, which is exactly the
robustness needed when the thing being measured is "did this explode"). Then
one further change: overfitting needed a *margin*, not just a direction,
because on any noisy curve validation sits fractionally above its best most of
the time.

11/20 → 15/20 → **18/20**, with false positives on healthy runs going to zero
in that sample. Tuning four thresholds separately would have moved the number
too, and left the cause in place.

---

## Tests

```bash
python test_tracker.py     # 21/21, about 2 seconds
```

| Group | Pins |
|---|---|
| **Durability** | 50 metrics survive `os._exit(1)`; batch mode loses at most `batch_size` and never reorders; a crash is recorded with its error; `__exit__` re-raises |
| **Correctness** | same `(key, step)` twice raises; NaN/±inf round-trip; step ordering; params keep their JSON type |
| **Artifacts** | copied not referenced — overwriting the original afterwards does not change the stored copy; sha256 recorded |
| **Detectors** | NaN reported at its *first* step; one spike is not divergence (regression); sustained growth is; healthy curves left alone; overfitting needs both curves |
| **End to end** | detection beats waiting on 15 labelled runs; delete removes everything; reads work mid-run (WAL) |

Two worth naming:

- **`test_nan_is_reported_at_the_step_it_first_appeared`** — once a loss is
  NaN every later value is NaN too. Reporting the last one overstates
  detection time by the entire rest of the run, and the compute-saved claim
  rests directly on that number.
- **`test_a_single_spike_is_not_divergence`** — the regression guard for the
  root cause above.

---

## What is not here

- **No PyTorch auto-logging hook.** A `Trainer` callback would need to guess
  at your loop's shape, and `run.log_metric(...)` is one line inside a loop
  you already wrote. `sklearn_run()` exists only because
  `estimator.get_params()` is genuinely the whole story for sklearn.
- **No Streamlit.** The CLI plus one generated HTML file covers it without a
  server or a package.
- **No PDF export.** `--export` writes CSV; the dashboard prints to PDF from
  a browser.
- **No distributed / multi-machine runs.** SQLite over a network filesystem
  is a bad idea, and saying so is better than shipping a footgun.
- **No hyperparameter search.** This records experiments; it does not choose
  them.
- **The detectors are thresholds, not a model.** Tuned on synthetic curves
  with a known shape. RESULTS.md §5 is explicit about what that does and does
  not transfer to a real training run.
