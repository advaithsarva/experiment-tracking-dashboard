# Results

Every number came from a command in this file, on this machine (Windows 11,
Python 3.12.1, standard library only). Everything is seeded; re-running any
command reproduces the figure exactly.

**Read section 5 before quoting anything here.** The curves are synthetic and
that decides what these numbers mean.

---

## 1. The test suite

```bash
python test_tracker.py
```

**21 / 21 passed**, about 2 seconds. No pytest, no network.

The load-bearing one is `test_metrics_survive_a_killed_process`. It writes a
child script, spawns it, logs 50 metrics, and kills the process with
`os._exit(1)` — no `atexit` handler, no flush, no `__exit__`, the hardest
death available — then reads the database from the parent.

**50 of 50 survive.** That is the project's central claim, tested the only way
it can honestly be tested: with an actual crash rather than an assertion that
a commit was issued.

`test_batch_mode_is_honest_about_what_it_loses` is its mirror. With
`batch_size=10` and 35 metrics logged before the same hard kill, between 30
and 35 survive — never fewer, never out of order. Batch mode is faster and
*can* lose the tail; the test pins exactly how much.

---

## 2. Durability costs 7.4×, and the trade is the point

```bash
python bench.py --only write
```

3,000 metrics per mode, identical work:

| Mode | Seconds | Metrics/sec | µs each | All values stored |
|---|---|---|---|---|
| **`commit`** (default) | 0.345 | **8,704** | 114.9 | yes |
| `batch(100)` | 0.048 | 62,184 | 16.1 | yes |
| `batch(500)` | 0.047 | **64,232** | 15.6 | yes |

Committing per metric is **7.4× slower** than the fastest batch mode.

It is still the default, because 8,704 metrics/sec is far more than any real
training loop produces — a step that logs 4 metrics could run at 2,000
steps/sec and never notice — and the tail of a crashed run is the reason the
tool exists. Speed that costs you the last 100 steps before an explosion is
not speed.

The benchmark verifies completeness in *both* modes. A fast writer that drops
values is not faster, it is broken.

---

## 3. Detection: 60 labelled runs

```bash
python simulate.py --runs 60 --steps 300
python bench.py --only detect
```

Ground truth comes from `simulate.py`, which assigns each run a pathology
before generating its curve. 12 runs per shape.

| Metric | Value |
|---|---|
| Exact accuracy (names the right failure) | **85.0%** |
| Problem detection (flags it as *anything*) | **100.0%** |
| False positives on healthy runs | **2 / 12 (16.7%)** |
| Mean compute saved vs running to the end | **55.1%** |
| Baseline "wait until the end" saves | **0.0%** |

Per shape:

| Shape | n | Exact | Flagged | Fires at | Compute saved |
|---|---|---|---|---|---|
| diverged | 12 | 66.7% | 12/12 | 45.2% | **54.8%** |
| healthy | 12 | 83.3% | 0/12 | — | — |
| overfitting | 12 | 100.0% | 12/12 | 55.0% | 45.0% |
| plateau | 12 | 100.0% | 12/12 | 58.9% | 41.1% |
| unstable | 12 | 75.0% | 12/12 | 20.4% | **79.6%** |

### The three things worth reading honestly

**The 16.7% false-positive rate is the real weakness.** Two of twelve healthy
runs get flagged. In practice that means roughly one in six good runs would be
interrupted by someone acting on this tool, which is exactly how detection
tooling gets switched off. It is reported here rather than buried because
`problem detection: 100%` on its own is a misleading headline — a detector
that flagged *every* run would also score 100% there.

**Exact accuracy (85%) is lower than problem detection (100%), and that gap is
benign.** Every pathological run is caught; some are given the wrong *name* —
`diverged` at 66.7% mostly means a diverging run was called `unstable`, which
it also genuinely is on the way up. The action ("stop this run") is identical.
If a single number is quoted, it should be 85%, not 100%.

**`unstable` fires at 20.4% and saves 79.6%, which is suspiciously good.** It
is: an unstable curve is recognisable almost immediately because the signal is
variance rather than trend. That also makes it the shape most likely to
produce a false positive on a merely noisy healthy run, and the two failure
numbers are two views of the same threshold.

---

## 4. Why "compute saved" is reported at all

Judging a finished curve is worthless — the run already cost what it cost. So
`diagnose_incrementally()` replays each run in `check_every=10` step
increments and returns the first point at which the evidence was sufficient.

| Approach | Detection | Compute saved |
|---|---|---|
| Wait until the end (baseline) | 100% by definition | **0%** |
| This detector | 100% | **55.1%** |

A detector that fired at 95% of the way through would score identically on the
first column and 5% on the second. That is why both are always reported.

---

## 5. What these numbers are not

**The curves are synthetic, and this is the caveat that matters most.**
`simulate.py` generates them as exponential decay towards a floor with
lognormal noise, plus a pathology injected at a randomised onset. That is what
makes ground truth possible, and it is also *cleaner than reality*:

- a real onset is gradual and often ambiguous; here it is a defined point
- real runs mix pathologies — overfitting *and* a plateau *and* an LR drop
- real curves have discontinuities from schedule changes, restarts and
  evaluation-set swaps, none of which are modelled
- the noise is stationary; real training noise shrinks as the LR decays

**So the detection rate transfers, and the exact numbers do not.** The rules
(NaN, sustained growth, curve separation, flat trend, variance vs descent) are
the same ones a human uses reading a TensorBoard plot. The thresholds — 3×
best, 2% margin, 40-point window — were tuned on these fixtures and would need
re-tuning on a real corpus of runs.

**There is no baseline of what is normal for your model.** A 3× loss spike is
divergence for a converged model and routine for the first 50 steps of a fresh
one. `MIN_POINTS = 12` is the only concession to that, and it is crude.

**`compute saved` assumes stopping is free and correct.** It counts steps
avoided if you killed the run at detection. It does not account for the cost
of being wrong — killing a healthy run 17% of the time — nor for runs that
recover on their own, which some plateaus do after an LR decay.

**The write benchmark is one machine, one SSD, one filesystem.** The 7.4×
ratio is stable in kind but not in magnitude; on a network filesystem the
committed mode would be far worse, which is one reason distributed runs are
listed as out of scope.
