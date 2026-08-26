"""Generate training runs with known outcomes, so the detector can be scored.

Every run is labelled by construction -- this generator decided whether it
diverges, overfits, plateaus or trains cleanly. That label is the ground truth
`bench.py` measures against. Without it the "detection rate" in RESULTS.md
would be a number with nothing behind it.

    python simulate.py --runs 60 --db runs.db
    python simulate.py --runs 60 --durability batch   # for the write benchmark

The curve shapes are the ones that actually show up in training, not arbitrary
maths: exponential decay towards a floor with multiplicative noise, plus
whichever pathology the run was assigned.
"""

import argparse
import math
import random

from tracker import Run

SHAPES = ["healthy", "diverged", "overfitting", "plateau", "unstable"]


def curve(shape, steps, rng, noise=0.03):
    """Return (train, val) as lists of (step, value).

    Base shape: loss = floor + (start - floor) * exp(-k * t), multiplied by
    lognormal noise. That is what a real loss curve looks like -- fast early
    progress, asymptotic approach to a floor, noise proportional to the value
    rather than additive.
    """
    start, floor = 2.4, 0.28
    k = 3.2 / steps
    train, val = [], []

    # Where a pathology kicks in. Randomised so the detector cannot learn a
    # fixed position, and returned implicitly through the curve itself.
    onset = rng.randint(int(steps * 0.25), int(steps * 0.6))

    for t in range(steps):
        base = floor + (start - floor) * math.exp(-k * t)
        wobble = math.exp(rng.gauss(0, noise))

        if shape == "healthy":
            tr = base * wobble
            va = (base * 1.06 + 0.02) * math.exp(rng.gauss(0, noise * 1.6))

        elif shape == "diverged":
            if t < onset:
                tr = base * wobble
                va = base * 1.06 * math.exp(rng.gauss(0, noise * 1.6))
            else:
                # Exponential blow-up, then NaN once it overflows -- which is
                # exactly how it goes in practice.
                blow = base * math.exp((t - onset) * 0.28)
                tr = float("nan") if blow > 1e12 else blow * wobble
                va = tr

        elif shape == "overfitting":
            tr = base * wobble
            if t < onset:
                va = base * 1.06 * math.exp(rng.gauss(0, noise * 1.6))
            else:
                # Train keeps falling, val turns and climbs steadily.
                climb = (t - onset) * (start - floor) * 0.006
                va = (floor + (start - floor) * math.exp(-k * onset) + climb) * \
                     math.exp(rng.gauss(0, noise * 1.6))

        elif shape == "plateau":
            frozen = min(t, onset)
            flat = floor + (start - floor) * math.exp(-k * frozen)
            tr = flat * math.exp(rng.gauss(0, noise * 0.25))
            va = flat * 1.06 * math.exp(rng.gauss(0, noise * 0.4))

        elif shape == "unstable":
            # Descends, but the step-to-step swing dwarfs the descent.
            swing = math.exp(rng.gauss(0, noise * 14))
            tr = base * swing
            va = base * 1.06 * math.exp(rng.gauss(0, noise * 14))

        else:
            raise ValueError(f"unknown shape {shape!r}")

        train.append((t, tr))
        if t % 5 == 0:                   # validation runs less often, as usual
            val.append((t, va))

    return train, val


def simulate(db_path="runs.db", n_runs=60, steps=300, seed=42, durability="commit",
             shapes=None):
    """Write `n_runs` runs to the database and return their ground-truth labels."""
    rng = random.Random(seed)
    shapes = shapes or SHAPES
    truth = []

    optimisers = ["adamw", "sgd", "adam"]
    for i in range(n_runs):
        shape = shapes[i % len(shapes)]
        lr = round(10 ** rng.uniform(-4.5, -2.0), 6)
        params = {
            "lr": lr,
            "batch_size": rng.choice([16, 32, 64, 128]),
            "optimizer": rng.choice(optimisers),
            "dropout": round(rng.uniform(0.0, 0.5), 2),
            "layers": rng.choice([2, 4, 6, 12]),
            "shape": shape,           # recorded so the dashboard can be checked by eye
        }
        train, val = curve(shape, steps, rng)

        with Run(f"{shape}-{i:03d}", params=params, db_path=db_path,
                 durability=durability, notes=f"synthetic run, shape={shape}") as run:
            for (step, tr) in train:
                run.log_metric("train_loss", tr, step)
            for (step, va) in val:
                run.log_metric("val_loss", va, step)
            run.log_metric("learning_rate", lr, 0)

            truth.append({"run_id": run.id, "name": run.name, "shape": shape,
                          "steps": steps, "lr": lr})

    return truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="runs.db")
    ap.add_argument("--runs", type=int, default=60)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--durability", choices=["commit", "batch"], default="commit")
    args = ap.parse_args()

    truth = simulate(args.db, args.runs, args.steps, args.seed, args.durability)
    counts = {}
    for row in truth:
        counts[row["shape"]] = counts.get(row["shape"], 0) + 1

    print(f"wrote {len(truth)} runs to {args.db} "
          f"({args.steps} steps each, durability={args.durability})")
    for shape, n in sorted(counts.items()):
        print(f"  {shape:<14}{n:>4}")


if __name__ == "__main__":
    main()
