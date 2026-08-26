"""Measure the two claims this project makes, against baselines that can win.

    python bench.py                  # both benchmarks
    python bench.py --only detect
    python bench.py --only write
    python bench.py --json

**Detection.** Ground truth comes from `simulate.py`, which decides each run's
pathology before generating its curve. Two numbers, and the second is the one
that matters:

    accuracy       did it name the right failure mode
    compute saved  what fraction of training steps killing the run would avoid

The baseline is "wait until the end", which by definition has perfect accuracy
and saves nothing. A detector that fires at 95% of the way through is no
better than that baseline and the compute-saved column says so.

**Write durability.** `durability="commit"` fsyncs a transaction per metric so
a crashed run keeps its tail. That costs throughput. This measures how much,
so the trade is a number rather than an opinion.
"""

import argparse
import json
import os
import random
import statistics
import tempfile
import time

import detect
import simulate
import tracker


def bench_detection(n_runs=60, steps=300, seed=42, check_every=10):
    db = os.path.join(tempfile.mkdtemp(prefix="bench-detect-"), "runs.db")
    truth = simulate.simulate(db, n_runs=n_runs, steps=steps, seed=seed)

    by_shape = {}
    exact = problems_found = problems_total = 0
    saved_fractions = []
    false_positives = 0
    healthy_total = 0

    for row in truth:
        metrics = tracker.get_metrics(row["run_id"], db_path=db)
        train = [(s, v) for s, v, _ in metrics.get("train_loss", [])]
        val = [(s, v) for s, v, _ in metrics.get("val_loss", [])]
        found, fraction = detect.diagnose_incrementally(train, val, check_every)

        shape = row["shape"]
        stats = by_shape.setdefault(shape, {"n": 0, "exact": 0, "flagged": 0,
                                            "fractions": []})
        stats["n"] += 1

        if shape == "healthy":
            healthy_total += 1
            if found.is_problem:
                false_positives += 1
            else:
                stats["exact"] += 1
                exact += 1
        else:
            problems_total += 1
            if found.is_problem:
                problems_found += 1
                stats["flagged"] += 1
                stats["fractions"].append(fraction)
                saved_fractions.append(1.0 - fraction)
            if found.kind == shape:
                stats["exact"] += 1
                exact += 1

    return {
        "runs": len(truth),
        "steps_per_run": steps,
        "exact_accuracy": round(exact / len(truth), 4),
        "problem_detection_rate": round(problems_found / problems_total, 4) if problems_total else 0,
        "false_positive_rate": round(false_positives / healthy_total, 4) if healthy_total else 0,
        "false_positives": false_positives,
        "healthy_runs": healthy_total,
        "mean_compute_saved": round(statistics.mean(saved_fractions), 4) if saved_fractions else 0,
        "median_compute_saved": round(statistics.median(saved_fractions), 4) if saved_fractions else 0,
        "baseline_compute_saved": 0.0,      # "wait until the end" saves nothing, by definition
        "by_shape": {
            k: {"n": v["n"],
                "exact_accuracy": round(v["exact"] / v["n"], 3),
                "flagged": v["flagged"],
                "mean_fired_at": round(statistics.mean(v["fractions"]), 3) if v["fractions"] else None,
                "mean_compute_saved": round(1 - statistics.mean(v["fractions"]), 3)
                if v["fractions"] else None}
            for k, v in sorted(by_shape.items())
        },
    }


def bench_writes(n_metrics=3000, seed=1):
    """Time both durability modes on identical work."""
    rng = random.Random(seed)
    values = [rng.gauss(1.0, 0.2) for _ in range(n_metrics)]
    out = {}

    for mode, batch in (("commit", 1), ("batch", 100), ("batch", 500)):
        directory = tempfile.mkdtemp(prefix=f"bench-write-{mode}{batch}-")
        db = os.path.join(directory, "runs.db")
        label = mode if mode == "commit" else f"batch({batch})"

        started = time.perf_counter()
        with tracker.Run("write-bench", db_path=db, durability=mode,
                         batch_size=batch) as run:
            for i, v in enumerate(values):
                run.log_metric("loss", v, i)
        elapsed = time.perf_counter() - started

        # Prove nothing was lost, in both modes. A fast writer that drops
        # values is not faster, it is broken.
        stored = tracker.get_metrics(run.id, db_path=db)["loss"]
        out[label] = {
            "seconds": round(elapsed, 3),
            "metrics_per_second": round(n_metrics / elapsed),
            "us_per_metric": round(elapsed / n_metrics * 1e6, 1),
            "values_stored": len(stored),
            "values_logged": n_metrics,
            "complete": len(stored) == n_metrics,
        }

    fastest = max(out.values(), key=lambda r: r["metrics_per_second"])
    out["commit"]["slowdown_vs_fastest"] = round(
        fastest["metrics_per_second"] / out["commit"]["metrics_per_second"], 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["detect", "write"])
    ap.add_argument("--runs", type=int, default=60)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--metrics", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    report = {}
    if args.only in (None, "detect"):
        report["detection"] = bench_detection(args.runs, args.steps, args.seed)
    if args.only in (None, "write"):
        report["writes"] = bench_writes(args.metrics, args.seed)

    if args.json:
        print(json.dumps(report, indent=2))
        return

    if "detection" in report:
        d = report["detection"]
        print(f"=== detection: {d['runs']} runs x {d['steps_per_run']} steps, seed {args.seed} ===\n")
        print(f"exact accuracy (names the right failure)   {d['exact_accuracy']:>8.1%}")
        print(f"problem detection (flags it as anything)   {d['problem_detection_rate']:>8.1%}")
        print(f"false positives on healthy runs            "
              f"{d['false_positives']:>4} / {d['healthy_runs']}   ({d['false_positive_rate']:.1%})")
        print(f"mean compute saved vs running to the end   {d['mean_compute_saved']:>8.1%}")
        print(f"  baseline 'wait until the end'            {d['baseline_compute_saved']:>8.1%}")

        print(f"\n{'shape':<14}{'n':>4}{'exact':>9}{'flagged':>9}{'fires at':>10}{'saves':>9}")
        for shape, r in d["by_shape"].items():
            fires = f"{r['mean_fired_at']:.1%}" if r["mean_fired_at"] is not None else "-"
            saves = f"{r['mean_compute_saved']:.1%}" if r["mean_compute_saved"] is not None else "-"
            print(f"{shape:<14}{r['n']:>4}{r['exact_accuracy']:>9.1%}"
                  f"{r['flagged']:>9}{fires:>10}{saves:>9}")

    if "writes" in report:
        w = report["writes"]
        print(f"\n=== writes: {args.metrics} metrics per mode ===\n")
        print(f"{'mode':<14}{'sec':>8}{'metrics/s':>12}{'us each':>10}{'complete':>10}")
        for mode, r in w.items():
            print(f"{mode:<14}{r['seconds']:>8.3f}{r['metrics_per_second']:>12,}"
                  f"{r['us_per_metric']:>10.1f}{'yes' if r['complete'] else 'NO':>10}")
        print(f"\ncommit mode is {w['commit']['slowdown_vs_fastest']}x slower than the "
              f"fastest batch mode, and is the default anyway: the tail of a crashed "
              f"run is the reason the tool exists.")


if __name__ == "__main__":
    main()
