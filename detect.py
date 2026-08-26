"""Spot a training run going wrong, early enough that killing it saves time.

The value proposition, stated so it can be tested
-------------------------------------------------
Anyone can tell a run failed by looking at the finished curve. The only useful
question is **how early**, because the point is to kill it and get the GPU
back. So every detector returns the step it fired at, and `bench.py` measures
two numbers against a "wait until the end" baseline:

    detection rate   -- of runs that really were pathological, how many were caught
    compute saved    -- of all training steps, how many were avoided

A detector that catches everything at 95% of the way through has a perfect
detection rate and saves nothing. Both numbers, always, or the first one is
marketing.

Four failure modes, in the order they cost you
----------------------------------------------
    diverged     loss is NaN, infinite, or exploding. Nothing is recoverable.
    overfitting  train keeps falling, val turns around. Recoverable, but the
                 best checkpoint is already behind you.
    plateau      no meaningful movement for a long stretch.
    unstable     loss oscillating far more than it is descending.

Each is a rule over the curve, not a model. That is deliberate and the
comparison in RESULTS.md is against the honest alternative (a human watching,
approximated by "wait until the end"), not against a strawman.
"""

import math
from dataclasses import dataclass

# A run must produce at least this many points before any judgement. Early
# training is genuinely noisy and firing at step 3 would make the tool
# untrustworthy on exactly the runs that were going to be fine.
MIN_POINTS = 12

DIVERGENCE_FACTOR = 3.0        # smoothed loss this many times its best -> exploding
DIVERGENCE_RUN = 3             # consecutive points above it, so one spike is not enough
PLATEAU_WINDOW = 40            # points of no movement before calling it
PLATEAU_TOLERANCE = 5e-3       # relative trend improvement below this is not movement
OVERFIT_PATIENCE = 4           # consecutive val increases while train keeps improving
OVERFIT_VAL_MARGIN = 0.02      # val must sit this far above its best, not merely above it
OVERFIT_TRAIN_MARGIN = 0.01    # and train must have improved this much since that best
INSTABILITY_RATIO = 2.0        # step-to-step swing vs net descent
SMOOTH_WINDOW = 5              # moving-median width; see _smooth


def _smooth(series, window=SMOOTH_WINDOW):
    """Moving median over `window` points.

    THE fix for this whole file. Every detector originally compared raw
    adjacent values, and real loss curves carry multiplicative noise -- so one
    lucky spike read as divergence, one unlucky dip reset the overfitting
    counter, and noise in a flat curve masked a plateau. Four wrong answers,
    one cause.

    A median rather than a mean: a single 1e12 spike destroys a mean and leaves
    a median untouched, which is exactly the robustness needed when the thing
    being measured is "did this explode".

    Non-finite values are dropped first; `_diverged` checks for them separately
    and earlier, because NaN is a certainty rather than a trend.
    """
    finite = _finite(series)
    if len(finite) < window:
        return finite
    out, half = [], window // 2
    for i in range(len(finite)):
        lo, hi = max(0, i - half), min(len(finite), i + half + 1)
        chunk = sorted(v for _, v in finite[lo:hi])
        out.append((finite[i][0], chunk[len(chunk) // 2]))
    return out


@dataclass
class Diagnosis:
    kind: str                  # healthy | diverged | overfitting | plateau | unstable
    step: int                  # the step the evidence became sufficient
    detail: str
    confidence: str            # certain | likely
    suggestion: str = ""

    @property
    def is_problem(self):
        return self.kind != "healthy"


def _finite(series):
    return [(s, v) for s, v in series if math.isfinite(v)]


def diagnose(train_curve, val_curve=None, total_steps=None):
    """Judge one run. `curve` is [(step, value), ...] in step order.

    Returns the FIRST problem in severity order, not a list. A diverged run is
    also technically plateaued once it hits infinity, and reporting both is
    noise -- the caller wants to know what to do, and there is only one answer.
    """
    if len(train_curve) < MIN_POINTS:
        return Diagnosis("healthy", train_curve[-1][0] if train_curve else 0,
                         f"only {len(train_curve)} points; too early to judge",
                         "certain")

    for check in (_diverged, _unstable, _overfitting, _plateau):
        found = check(train_curve, val_curve)
        if found:
            return found

    last = train_curve[-1]
    return Diagnosis("healthy", last[0],
                     f"loss {last[1]:.4g} at step {last[0]}, still descending", "certain")


def _diverged(train_curve, val_curve=None):
    """NaN, infinity, or loss climbing well above its own best.

    NaN is checked first and reported at the step it *first* appears. Once a
    loss is NaN every subsequent value is NaN too, so reporting the last step
    would overstate how long it took to notice by the entire rest of the run.
    """
    for step, value in train_curve:
        if math.isnan(value):
            return Diagnosis("diverged", step, f"loss became NaN at step {step}",
                             "certain",
                             "Lower the learning rate, add gradient clipping, and check "
                             "for a log/divide of a zero or negative value.")
        if math.isinf(value):
            return Diagnosis("diverged", step, f"loss became infinite at step {step}",
                             "certain", "Lower the learning rate and clip gradients.")

    smoothed = _smooth(train_curve)
    if len(smoothed) < MIN_POINTS:
        return None

    # Sustained, not instantaneous. A noisy-but-descending run throws single
    # points well above its best constantly; a diverging one stays up. This is
    # what stopped unstable runs being misreported as diverged.
    best, consecutive = smoothed[0][1], 0
    for step, value in smoothed:
        if value < best:
            best = value
        # Guard against a best near 0: a tiny denominator makes noise look
        # like an explosion.
        if best > 1e-8 and value > best * DIVERGENCE_FACTOR:
            consecutive += 1
            if consecutive >= DIVERGENCE_RUN:
                return Diagnosis(
                    "diverged", step,
                    f"smoothed loss {value:.4g} at step {step} is {value / best:.1f}x "
                    f"its best ({best:.4g}), for {consecutive} points running",
                    "certain",
                    "The learning rate is almost certainly too high. Warmup or "
                    "clipping would likely fix it.")
        else:
            consecutive = 0
    return None


def _unstable(train_curve, val_curve=None):
    """Oscillating much more than it is descending.

    Compares mean absolute step-to-step change against total net descent over
    the same window. A healthy curve descends more than it wobbles; an
    unstable one wobbles more than it descends and may still trend downwards,
    which is why net descent alone would miss it.
    """
    finite = _finite(train_curve)
    if len(finite) < MIN_POINTS * 2:
        return None

    window = finite[len(finite) // 2:]         # ignore the noisy early phase
    values = [v for _, v in window]
    swings = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
    if not swings:
        return None

    mean_swing = sum(swings) / len(swings)
    # Net descent from the SMOOTHED curve. Taking it from raw endpoints makes
    # the measurement depend on whether the last point was a spike -- which is
    # the very noise being measured.
    sm = _smooth(window)
    net_descent = (sm[0][1] - sm[-1][1]) if len(sm) >= 2 else 0.0

    if net_descent > 0 and mean_swing > net_descent * INSTABILITY_RATIO:
        return Diagnosis(
            "unstable", window[-1][0],
            f"mean step-to-step swing {mean_swing:.4g} is "
            f"{mean_swing / net_descent:.1f}x the net descent {net_descent:.4g}",
            "likely",
            "Reduce the learning rate or raise the batch size; the gradient estimate "
            "is noisier than the signal it is following.")
    return None


def _overfitting(train_curve, val_curve=None):
    """Train falling while validation climbs, for several evaluations running.

    Requires BOTH conditions. Validation rising on its own is often noise; the
    diagnostic signal is the two curves separating, which is what
    overfitting actually is.
    """
    if not val_curve or len(val_curve) < MIN_POINTS:
        return None

    val = _smooth(val_curve, 3)                # val is logged less often: smaller window
    train = dict(_smooth(train_curve))
    if len(val) < MIN_POINTS:
        return None

    best_val, best_step = val[0][1], val[0][0]
    rising = 0
    for i in range(1, len(val)):
        step, value = val[i]
        if value < best_val:
            best_val, best_step, rising = value, step, 0
            continue

        # Both sides need a MARGIN, not just a direction.
        #
        # Direction alone made this the greediest detector in the file: on any
        # noisy curve validation sits fractionally above its best most of the
        # time, and a descending train curve is always below its own earlier
        # best. That combination fired on healthy, plateaued and unstable runs
        # alike -- four false positives out of twenty. Requiring validation to
        # be 2% above its low-water mark and training to have gained 1% since
        # then is what separates "the curves have genuinely separated" from
        # "the curves wobbled".
        val_risen = best_val > 0 and (value - best_val) / abs(best_val) > OVERFIT_VAL_MARGIN

        train_now = train.get(step)
        earlier = [v for s2, v in train.items() if s2 <= best_step]
        train_gained = (
            train_now is not None and bool(earlier)
            and min(earlier) > 0
            and (min(earlier) - train_now) / abs(min(earlier)) > OVERFIT_TRAIN_MARGIN
        )

        rising = rising + 1 if (val_risen and train_gained) else 0

        if rising >= OVERFIT_PATIENCE:
            return Diagnosis(
                "overfitting", step,
                f"validation has sat >{OVERFIT_VAL_MARGIN:.0%} above its best for "
                f"{rising} evaluations while training loss kept improving; best "
                f"validation was {best_val:.4g} at step {best_step}",
                "likely",
                f"Stop and take the step {best_step} checkpoint. Then add "
                f"regularisation (dropout, weight decay) or reduce model size.")
    return None


def _plateau(train_curve, val_curve=None):
    """No meaningful relative improvement for PLATEAU_WINDOW steps."""
    finite = _finite(train_curve)
    if len(finite) < PLATEAU_WINDOW:
        return None

    # Mean of the window's first half against its second half. The original
    # took min() against the first point, so on a flat-but-noisy curve a single
    # lucky dip counted as improvement and no plateau was ever reported. A
    # half-to-half trend is immune to that.
    window = _smooth(finite)[-PLATEAU_WINDOW:]
    if len(window) < PLATEAU_WINDOW // 2:
        return None
    values = [v for _, v in window]
    half = len(values) // 2
    first = sum(values[:half]) / half
    second = sum(values[half:]) / (len(values) - half)
    if abs(first) < 1e-12:
        return None

    improvement = (first - second) / abs(first)
    if improvement < PLATEAU_TOLERANCE:
        return Diagnosis(
            "plateau", window[-1][0],
            f"relative trend over the last {PLATEAU_WINDOW} points is "
            f"{improvement:.2e}, below {PLATEAU_TOLERANCE:.0e}",
            "likely",
            "Decay the learning rate, or stop -- this run has converged as far as "
            "this configuration will take it.")
    return None


def diagnose_incrementally(train_curve, val_curve=None, check_every=10):
    """Replay the run and report the FIRST step a problem became detectable.

    This is what makes the value claim measurable. Judging a finished curve
    tells you a run failed; replaying it tells you when you could have known,
    and therefore how much compute killing it would have saved.

    Returns (Diagnosis, fraction_of_run_elapsed) or (healthy, 1.0).
    """
    if not train_curve:
        return Diagnosis("healthy", 0, "no data", "certain"), 1.0

    total = train_curve[-1][0] or 1
    val_by_step = dict(val_curve or [])

    for i in range(MIN_POINTS, len(train_curve) + 1, check_every):
        partial_train = train_curve[:i]
        cutoff = partial_train[-1][0]
        partial_val = [(s, v) for s, v in (val_curve or []) if s <= cutoff]

        found = diagnose(partial_train, partial_val or None)
        if found.is_problem:
            return found, min(1.0, cutoff / total)

    return diagnose(train_curve, val_curve), 1.0
