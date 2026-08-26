"""Plain asserts, no pytest, no network. About two seconds.

    python test_tracker.py

The first group is the durability invariant, checked by actually killing a
subprocess mid-run rather than by trusting that a commit happened. Everything
else in this project is built on the claim that a logged value survives a
crash, so that claim gets tested the only way it can be: with a crash.
"""

import math
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

import detect
import simulate
import tracker

TMP = tempfile.mkdtemp(prefix="tracker-tests-")
results = []


def check(name, fn):
    try:
        fn()
    except AssertionError as exc:
        results.append((name, False, str(exc)))
    except Exception as exc:
        results.append((name, False, f"{type(exc).__name__}: {exc}"))
    else:
        results.append((name, True, ""))


def fresh(name):
    return os.path.join(TMP, f"{name}.db")


# --------------------------------------------------------------------------
# the invariant: a logged value survives the process that logged it
# --------------------------------------------------------------------------

def test_metrics_survive_a_killed_process():
    """THE test. Spawn a child, log 50 metrics, kill it with os._exit(1) so no
    atexit handler, no flush, no __exit__ ever runs. All 50 must be readable.

    This is the entire premise of an experiment tracker: you are logging so
    that when a 6-hour run dies at hour 5 you can see what happened before it
    died. A tracker that buffers in memory loses exactly the tail that
    explains the crash.
    """
    db = fresh("crash")
    script = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
        from tracker import Run
        run = Run("doomed", db_path={db!r}, run_id="deadbeef0001")
        for i in range(50):
            run.log_metric("loss", 1.0 / (i + 1), i)
        os._exit(1)          # hardest possible death: no cleanup of any kind
    """)
    path = os.path.join(TMP, "crash_child.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(script)

    proc = subprocess.run([sys.executable, path], capture_output=True, text=True)
    assert proc.returncode == 1, f"child exited {proc.returncode}: {proc.stderr[:300]}"

    stored = tracker.get_metrics("deadbeef0001", db_path=db).get("loss", [])
    assert len(stored) == 50, (
        f"only {len(stored)} of 50 metrics survived os._exit -- the tail of a "
        f"crashed run is exactly what this tool exists to keep"
    )
    assert stored[-1][0] == 49, f"last step is {stored[-1][0]}, expected 49"


def test_batch_mode_is_honest_about_what_it_loses():
    """The other half. `durability='batch'` is faster and CAN lose the tail --
    that is the documented trade. What it must not do is lose values it
    already flushed, and it must never silently reorder."""
    db = fresh("batchcrash")
    script = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
        from tracker import Run
        run = Run("batched", db_path={db!r}, run_id="deadbeef0002",
                  durability="batch", batch_size=10)
        for i in range(35):
            run.log_metric("loss", float(i), i)
        os._exit(1)
    """)
    path = os.path.join(TMP, "batch_child.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(script)
    subprocess.run([sys.executable, path], capture_output=True, text=True)

    stored = tracker.get_metrics("deadbeef0002", db_path=db).get("loss", [])
    # 35 logged, batch of 10 -> 30 flushed, up to 10 lost. Never more.
    assert 30 <= len(stored) <= 35, f"batch mode kept {len(stored)}, expected 30-35"
    assert [s for s, _, _ in stored] == sorted(s for s, _, _ in stored), \
        "batch mode wrote steps out of order"


def test_a_crashed_run_is_marked_crashed_and_says_why():
    """Without this the dashboard cannot tell 'still running' from 'died three
    days ago', which is the question people ask most."""
    db = fresh("status")
    try:
        with tracker.Run("will-fail", db_path=db) as run:
            run.log_metric("loss", 1.0, 0)
            raise RuntimeError("synthetic explosion")
    except RuntimeError:
        pass

    found = tracker.get_run(run.id, db)
    assert found["status"] == "crashed", f"status is {found['status']!r}"
    assert "synthetic explosion" in (found["error"] or ""), \
        f"the error was not recorded: {found['error']!r}"
    assert tracker.get_metrics(run.id, db_path=db)["loss"], \
        "metrics logged before the crash were lost"


def test_the_context_manager_never_swallows_the_exception():
    """Recording the failure must not hide it from the caller."""
    db = fresh("reraise")
    try:
        with tracker.Run("reraise", db_path=db):
            raise ValueError("must propagate")
    except ValueError:
        pass
    else:
        raise AssertionError("__exit__ swallowed the exception")


# --------------------------------------------------------------------------
# ordering and NaN
# --------------------------------------------------------------------------

def test_logging_the_same_step_twice_raises():
    """Overwriting would make a loss curve that dropped and then rose look
    monotonic -- the single most misleading thing a tracker can do."""
    db = fresh("dupe")
    with tracker.Run("dupes", db_path=db) as run:
        run.log_metric("loss", 1.0, 5)
        try:
            run.log_metric("loss", 99.0, 5)
            raise AssertionError("the same (key, step) was accepted twice")
        except tracker.TrackerError as exc:
            assert "5" in str(exc)


def test_nan_and_infinity_round_trip():
    """A NaN loss is the most diagnostic value a tracker can hold. SQLite REAL
    turns NaN into NULL, which converts 'the loss went NaN at step 2' into
    'the curve just stops' -- losing the explanation entirely."""
    db = fresh("nan")
    with tracker.Run("nans", db_path=db) as run:
        run.log_metric("loss", 1.5, 0)
        run.log_metric("loss", float("nan"), 1)
        run.log_metric("loss", float("inf"), 2)
        run.log_metric("loss", float("-inf"), 3)

    values = [v for _, v, _ in tracker.get_metrics(run.id, db_path=db)["loss"]]
    assert len(values) == 4, f"{len(values)} of 4 values survived"
    assert values[0] == 1.5
    assert math.isnan(values[1]), f"NaN came back as {values[1]!r}"
    assert values[2] == float("inf"), f"inf came back as {values[2]!r}"
    assert values[3] == float("-inf"), f"-inf came back as {values[3]!r}"


def test_metrics_come_back_in_step_order():
    """Logged out of order on purpose. A curve is only a curve in order."""
    db = fresh("order")
    with tracker.Run("shuffled", db_path=db) as run:
        for step in (7, 2, 9, 0, 4):
            run.log_metric("loss", float(step), step)
    steps = [s for s, _, _ in tracker.get_metrics(run.id, db_path=db)["loss"]]
    assert steps == [0, 2, 4, 7, 9], f"got {steps}"


def test_params_keep_their_type():
    """Coercing everything to float loses optimiser names and layer lists;
    storing repr() makes them unparseable later. JSON keeps the type."""
    db = fresh("params")
    with tracker.Run("typed", params={
        "lr": 3e-4, "optimizer": "adamw", "layers": [64, 32],
        "use_amp": True, "note": None,
    }, db_path=db) as run:
        pass
    got = tracker.get_run(run.id, db)["params"]
    assert got["lr"] == 3e-4 and isinstance(got["lr"], float)
    assert got["optimizer"] == "adamw"
    assert got["layers"] == [64, 32], f"list became {got['layers']!r}"
    assert got["use_amp"] is True, f"bool became {got['use_amp']!r}"


def test_a_negative_step_and_an_empty_name_are_refused():
    db = fresh("bad")
    try:
        tracker.Run("   ", db_path=db)
        raise AssertionError("a blank run name was accepted")
    except tracker.TrackerError:
        pass
    with tracker.Run("ok", db_path=db) as run:
        try:
            run.log_metric("loss", 1.0, -1)
            raise AssertionError("a negative step was accepted")
        except tracker.TrackerError:
            pass
    try:
        tracker.Run("x", db_path=db, durability="maybe")
        raise AssertionError("an unknown durability mode was accepted")
    except tracker.TrackerError:
        pass


# --------------------------------------------------------------------------
# artifacts
# --------------------------------------------------------------------------

def test_artifacts_are_copied_and_hashed():
    """Recorded by reference, a path points at a file someone overwrites next
    week, and the run's record quietly describes a different model. The hash
    is what makes 'is this the checkpoint that produced these numbers?'
    answerable."""
    db = fresh("artifacts")
    source = os.path.join(TMP, "model.bin")
    with open(source, "wb") as fh:
        fh.write(b"weights v1")

    with tracker.Run("artifact-run", db_path=db) as run:
        stored_path = run.log_artifact(source)

    with open(source, "wb") as fh:          # overwrite the original afterwards
        fh.write(b"COMPLETELY DIFFERENT")

    rows = tracker.get_artifacts(run.id, db)
    assert len(rows) == 1, f"{len(rows)} artifacts recorded"
    with open(stored_path, "rb") as fh:
        assert fh.read() == b"weights v1", \
            "the stored artifact changed when the original was overwritten"
    assert len(rows[0]["sha256"]) == 64
    assert rows[0]["bytes"] == 10

    try:
        with tracker.Run("missing", db_path=db) as run2:
            run2.log_artifact(os.path.join(TMP, "nope.bin"))
        raise AssertionError("a missing artifact file was accepted")
    except tracker.TrackerError:
        pass


# --------------------------------------------------------------------------
# the detectors
# --------------------------------------------------------------------------

def test_nan_is_reported_at_the_step_it_first_appeared():
    """Once a loss is NaN every later value is NaN too. Reporting the last one
    overstates how long detection took by the entire rest of the run -- which
    is the number the whole compute-saved claim rests on."""
    curve = [(i, 1.0 / (i + 1)) for i in range(20)] + \
            [(i, float("nan")) for i in range(20, 200)]
    found = detect.diagnose(curve)
    assert found.kind == "diverged", f"got {found.kind}"
    assert found.step == 20, f"reported step {found.step}, NaN began at 20"


def test_a_single_spike_is_not_divergence():
    """REGRESSION. The detector originally fired on any single point above 3x
    its best, so every noisy-but-descending run was reported as diverged. This
    is the bug the moving median was added to fix."""
    curve = [(i, 2.0 * math.exp(-i * 0.02)) for i in range(150)]
    curve[70] = (70, curve[70][1] * 12)          # one enormous spike
    found = detect.diagnose(curve)
    assert found.kind != "diverged", \
        f"one spike was reported as {found.kind} with detail: {found.detail}"


def test_sustained_growth_is_divergence():
    """The other side of the same rule: it must still catch a real explosion
    that never produces NaN."""
    curve = [(i, 2.0 * math.exp(-i * 0.02)) for i in range(60)] + \
            [(i, 1.0 * math.exp((i - 60) * 0.15)) for i in range(60, 150)]
    found = detect.diagnose(curve)
    assert found.kind == "diverged", f"a 90-step exponential blow-up read as {found.kind}"


def test_a_healthy_curve_is_left_alone():
    """False positives are what get a detector switched off."""
    import random
    rng = random.Random(0)
    for trial in range(12):
        train, val = simulate.curve("healthy", 300, rng)
        found = detect.diagnose(train, val)
        assert not found.is_problem, \
            f"trial {trial}: healthy run flagged as {found.kind} -- {found.detail}"


def test_overfitting_needs_both_curves_to_separate():
    """Validation rising alone is usually noise. The diagnostic signal is the
    two curves separating, which is what overfitting actually is -- so a run
    where BOTH curves plateau must not be called overfitting."""
    train = [(i, 1.0 + 0.0001 * i) for i in range(200)]     # train NOT improving
    val = [(i, 1.0 + 0.01 * i) for i in range(0, 200, 5)]   # val clearly rising
    found = detect.diagnose(train, val)
    assert found.kind != "overfitting", \
        f"called it overfitting when training loss was not improving: {found.detail}"


def test_diagnose_incrementally_fires_before_the_end():
    """The whole value claim. Judging a finished curve says a run failed;
    replaying it says when you could have known and therefore what killing it
    would have saved."""
    import random
    rng = random.Random(3)
    train, val = simulate.curve("diverged", 300, rng)
    found, fraction = detect.diagnose_incrementally(train, val)
    assert found.kind == "diverged", f"got {found.kind}"
    assert fraction < 0.95, f"only detectable at {fraction:.0%} of the run -- saves nothing"


def test_too_few_points_is_not_a_diagnosis():
    """Firing at step 3 would make the tool untrustworthy on exactly the runs
    that were going to be fine."""
    assert not detect.diagnose([(i, 1.0) for i in range(5)]).is_problem
    assert not detect.diagnose([]).is_problem, "an empty curve was diagnosed"


def test_every_problem_carries_a_suggestion():
    """A diagnosis with no action is an observation."""
    import random
    rng = random.Random(11)
    for shape in ("diverged", "overfitting", "plateau", "unstable"):
        train, val = simulate.curve(shape, 300, rng)
        found = detect.diagnose(train, val)
        if found.is_problem:
            assert found.suggestion, f"{shape} -> {found.kind} with no suggestion"
            assert found.confidence in ("certain", "likely"), \
                f"{found.kind} has confidence {found.confidence!r}"


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------

def test_the_detector_beats_waiting_on_ground_truth():
    """The measured claim, at small scale so the suite stays fast. bench.py
    runs the full version."""
    db = fresh("bench")
    truth = simulate.simulate(db, n_runs=15, steps=200, seed=7)

    flagged = saved = problems = false_positives = healthy = 0
    for row in truth:
        metrics = tracker.get_metrics(row["run_id"], db_path=db)
        train = [(s, v) for s, v, _ in metrics.get("train_loss", [])]
        val = [(s, v) for s, v, _ in metrics.get("val_loss", [])]
        found, fraction = detect.diagnose_incrementally(train, val)

        if row["shape"] == "healthy":
            healthy += 1
            false_positives += found.is_problem
        else:
            problems += 1
            if found.is_problem:
                flagged += 1
                saved += 1 - fraction

    assert flagged == problems, f"detected {flagged} of {problems} pathological runs"
    assert false_positives <= healthy * 0.5, \
        f"{false_positives} of {healthy} healthy runs flagged -- too noisy to trust"
    assert saved / flagged > 0.2, \
        f"mean compute saved {saved / flagged:.1%}; the baseline saves 0% and this is close to it"


def test_deleting_a_run_removes_everything():
    db = fresh("delete")
    source = os.path.join(TMP, "art.bin")
    with open(source, "wb") as fh:
        fh.write(b"x")
    with tracker.Run("doomed", params={"a": 1}, db_path=db) as run:
        run.log_metric("loss", 1.0, 0)
        run.log_artifact(source)

    tracker.delete_run(run.id, db)
    assert tracker.get_run(run.id, db) is None, "the run row survived"
    assert not tracker.get_metrics(run.id, db_path=db), "metrics survived"
    assert not tracker.get_artifacts(run.id, db), "artifact rows survived"


def test_reads_work_while_a_run_is_still_writing():
    """WAL mode. The first thing anyone does with a tracker is watch a live
    run; without WAL the reader blocks the writer."""
    db = fresh("concurrent")
    with tracker.Run("live", db_path=db) as run:
        for i in range(20):
            run.log_metric("loss", float(i), i)
            if i == 10:
                seen = tracker.get_metrics(run.id, db_path=db).get("loss", [])
                assert len(seen) == 11, \
                    f"mid-run read saw {len(seen)} of 11 values already committed"


TESTS = [
    ("metrics survive a killed process", test_metrics_survive_a_killed_process),
    ("batch mode is honest about what it loses", test_batch_mode_is_honest_about_what_it_loses),
    ("a crashed run is marked crashed", test_a_crashed_run_is_marked_crashed_and_says_why),
    ("the context manager re-raises", test_the_context_manager_never_swallows_the_exception),
    ("logging the same step twice raises", test_logging_the_same_step_twice_raises),
    ("nan and infinity round trip", test_nan_and_infinity_round_trip),
    ("metrics come back in step order", test_metrics_come_back_in_step_order),
    ("params keep their type", test_params_keep_their_type),
    ("bad input is refused", test_a_negative_step_and_an_empty_name_are_refused),
    ("artifacts are copied and hashed", test_artifacts_are_copied_and_hashed),
    ("nan is reported at its first step", test_nan_is_reported_at_the_step_it_first_appeared),
    ("a single spike is not divergence", test_a_single_spike_is_not_divergence),
    ("sustained growth is divergence", test_sustained_growth_is_divergence),
    ("a healthy curve is left alone", test_a_healthy_curve_is_left_alone),
    ("overfitting needs both curves", test_overfitting_needs_both_curves_to_separate),
    ("incremental diagnosis fires early", test_diagnose_incrementally_fires_before_the_end),
    ("too few points is not a diagnosis", test_too_few_points_is_not_a_diagnosis),
    ("every problem carries a suggestion", test_every_problem_carries_a_suggestion),
    ("the detector beats waiting", test_the_detector_beats_waiting_on_ground_truth),
    ("deleting a run removes everything", test_deleting_a_run_removes_everything),
    ("reads work during a live run", test_reads_work_while_a_run_is_still_writing),
]


def main():
    for name, fn in TESTS:
        check(name, fn)
    passed = sum(1 for _, ok, _ in results if ok)
    for name, ok, err in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            print(f"      {err}")
    print(f"\n{passed}/{len(results)} passed")
    shutil.rmtree(TMP, ignore_errors=True)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
