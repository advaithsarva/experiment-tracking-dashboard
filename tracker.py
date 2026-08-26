"""Experiment tracking on SQLite. The logging SDK and the storage it writes to.

THE INVARIANT
-------------
**A value that `log_metric` returned from is on disk. If the process dies one
instruction later, that value survives.**

This is the whole reason an experiment tracker exists. You are logging
precisely so that when a 6-hour training run dies at hour 5 you can see what
happened before it died. A tracker that buffers metrics in memory and flushes
"periodically" loses exactly the tail you need -- the last few steps before
the crash, which are the ones that explain it.

So the default is `durability="commit"`: every `log_metric` is its own
committed transaction. That is slower, and RESULTS.md measures by how much
(it is not close). `durability="batch"` is available for tight inner loops and
is honest about what it costs you: a crash loses up to `batch_size` values.

Ordering is the other half. Metrics are `(run_id, key, step, value, wall_time)`
and steps are unique per `(run, key)` -- logging the same step twice is a
programming error, not something to silently overwrite. An overwrite would
make a loss curve that dropped and then rose look monotonic.

    from tracker import Run

    with Run("resnet-lr-sweep", params={"lr": 3e-4, "batch": 64}) as run:
        for step in range(1000):
            run.log_metric("train_loss", loss, step)
            run.log_metric("val_loss", val, step)
        run.log_artifact("model.pt")
"""

import hashlib
import json
import os
import shutil
import sqlite3
import time
import uuid
from contextlib import contextmanager

DEFAULT_DB = os.environ.get("TRACKER_DB", "runs.db")
ARTIFACT_DIR = os.environ.get("TRACKER_ARTIFACTS", "artifacts")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'running',
    started     REAL NOT NULL,
    ended       REAL,
    error       TEXT,
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS params (
    run_id      TEXT NOT NULL REFERENCES runs(id),
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    PRIMARY KEY (run_id, key)
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id      TEXT NOT NULL REFERENCES runs(id),
    key         TEXT NOT NULL,
    step        INTEGER NOT NULL,
    value       REAL NOT NULL,
    wall_time   REAL NOT NULL,
    PRIMARY KEY (run_id, key, step)
);

CREATE TABLE IF NOT EXISTS artifacts (
    run_id      TEXT NOT NULL REFERENCES runs(id),
    name        TEXT NOT NULL,
    path        TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    bytes       INTEGER NOT NULL,
    logged      REAL NOT NULL,
    PRIMARY KEY (run_id, name)
);

CREATE INDEX IF NOT EXISTS metrics_by_run ON metrics(run_id, key, step);
CREATE INDEX IF NOT EXISTS runs_by_started ON runs(started DESC);
"""


class TrackerError(Exception):
    """Something the caller can fix. Never raised for a storage problem --
    those propagate as sqlite3 errors, because silently swallowing a write
    failure defeats the entire point of the tool."""


def connect(db_path=DEFAULT_DB):
    """Open the database, creating it if needed.

    WAL mode so a dashboard can read while a run is still writing. Without it
    the reader blocks the writer, and the first thing anyone does with a
    tracker is watch a live run.
    """
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    # NORMAL rather than FULL: WAL + NORMAL survives a process crash, which is
    # the failure this tool is for. Only a machine power-cut can lose the tail,
    # and paying an fsync per metric to cover that is not the right trade.
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


class Run:
    """One training run. Use as a context manager so status is always correct.

    The context manager is not sugar. Without it, a crashed run stays marked
    `running` forever and the dashboard cannot tell "still going" from "died
    three days ago" -- and "which of my runs died" is a question people ask
    constantly.
    """

    def __init__(self, name, params=None, db_path=DEFAULT_DB, run_id=None,
                 durability="commit", batch_size=100, notes=""):
        if durability not in ("commit", "batch"):
            raise TrackerError(
                f"durability must be 'commit' or 'batch', got {durability!r}"
            )
        if not name or not name.strip():
            raise TrackerError("a run needs a name; unnamed runs are unfindable")

        self.name = name.strip()
        self.id = run_id or uuid.uuid4().hex[:12]
        self.db_path = db_path
        self.durability = durability
        self.batch_size = batch_size
        self.conn = connect(db_path)
        self._pending = []
        self._seen = set()          # (key, step) already logged, for the ordering check

        self.conn.execute(
            "INSERT INTO runs (id, name, status, started, notes) VALUES (?,?,?,?,?)",
            (self.id, self.name, "running", time.time(), notes),
        )
        self.conn.commit()

        if params:
            self.log_params(params)

    # -- parameters ------------------------------------------------------

    def log_param(self, key, value):
        """Parameters are stored as JSON text, not floats.

        A learning rate is a float, an optimiser name is a string, a layer
        schedule is a list. Coercing everything to float loses the last two;
        storing repr() makes them un-parseable later. JSON keeps the type.
        """
        self.conn.execute(
            "INSERT OR REPLACE INTO params (run_id, key, value) VALUES (?,?,?)",
            (self.id, str(key), json.dumps(value, default=str)),
        )
        self.conn.commit()

    def log_params(self, mapping):
        for key, value in mapping.items():
            self.log_param(key, value)

    # -- metrics ---------------------------------------------------------

    def log_metric(self, key, value, step):
        """Record one number. Returns only after it is durable (default mode).

        A non-finite value is stored, not rejected. NaN loss is the single most
        important thing a tracker can show you, and a tracker that drops it
        leaves a gap in the curve exactly where the explanation is. SQLite
        cannot store NaN as REAL, so it goes in as a sentinel and comes back
        out as float('nan') -- see `_encode`/`_decode`.
        """
        step = int(step)
        if step < 0:
            raise TrackerError(f"step must be >= 0, got {step}")

        if (key, step) in self._seen:
            raise TrackerError(
                f"{key!r} step {step} was already logged for this run. "
                f"Overwriting would hide a curve that went back up; use a new "
                f"step or a different key."
            )
        self._seen.add((key, step))

        row = (self.id, str(key), step, _encode(value), time.time())
        if self.durability == "commit":
            self.conn.execute(
                "INSERT INTO metrics (run_id, key, step, value, wall_time) VALUES (?,?,?,?,?)",
                row,
            )
            self.conn.commit()
        else:
            self._pending.append(row)
            if len(self._pending) >= self.batch_size:
                self.flush()

    def log_metrics(self, mapping, step):
        for key, value in mapping.items():
            self.log_metric(key, value, step)

    def flush(self):
        """Write anything buffered. A no-op in 'commit' mode."""
        if not self._pending:
            return
        self.conn.executemany(
            "INSERT INTO metrics (run_id, key, step, value, wall_time) VALUES (?,?,?,?,?)",
            self._pending,
        )
        self.conn.commit()
        self._pending.clear()

    # -- artifacts -------------------------------------------------------

    def log_artifact(self, path, name=None):
        """Copy a file into the artifact store and record its hash.

        Copied, not referenced. A path recorded today points at a file someone
        overwrites next week, and then the run's record quietly describes a
        different model. The sha256 is what makes "is this the checkpoint that
        produced these numbers?" answerable.
        """
        if not os.path.isfile(path):
            raise TrackerError(f"no such artifact file: {path}")

        name = name or os.path.basename(path)
        digest = _sha256(path)
        target_dir = os.path.join(ARTIFACT_DIR, self.id)
        os.makedirs(target_dir, exist_ok=True)
        target = os.path.join(target_dir, name)
        shutil.copy2(path, target)

        self.conn.execute(
            "INSERT OR REPLACE INTO artifacts (run_id, name, path, sha256, bytes, logged)"
            " VALUES (?,?,?,?,?,?)",
            (self.id, name, target, digest, os.path.getsize(target), time.time()),
        )
        self.conn.commit()
        return target

    # -- lifecycle -------------------------------------------------------

    def finish(self, status="finished", error=None):
        self.flush()
        self.conn.execute(
            "UPDATE runs SET status=?, ended=?, error=? WHERE id=?",
            (status, time.time(), error, self.id),
        )
        self.conn.commit()
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.finish("finished")
        else:
            # Record the failure rather than losing it. The metrics logged
            # before the exception are already durable; this makes the run
            # findable as the one that crashed and says why.
            self.finish("crashed", f"{exc_type.__name__}: {exc}")
        return False        # never swallow the exception


# --------------------------------------------------------------------------
# NaN and infinity round-tripping
# --------------------------------------------------------------------------

_NAN = -1.7976931348623157e308      # sentinels, distinguishable from real data
_POSINF = 1.7976931348623157e308
_NEGINF = -1.7976931348623156e308


def _encode(value):
    """SQLite REAL cannot hold NaN -- it silently becomes NULL.

    A NaN loss is the most diagnostic value a tracker can capture, so it must
    survive the round trip. Storing it as NULL loses the fact that a value was
    logged at all, which turns 'the loss went NaN at step 400' into 'the curve
    just stops'.
    """
    value = float(value)
    if value != value:
        return _NAN
    if value == float("inf"):
        return _POSINF
    if value == float("-inf"):
        return _NEGINF
    return value


def _decode(value):
    if value == _NAN:
        return float("nan")
    if value == _POSINF:
        return float("inf")
    if value == _NEGINF:
        return float("-inf")
    return value


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# reading it back
# --------------------------------------------------------------------------

def list_runs(db_path=DEFAULT_DB, limit=200, status=None, name_like=None):
    conn = connect(db_path)
    sql = "SELECT * FROM runs"
    where, args = [], []
    if status:
        where.append("status = ?")
        args.append(status)
    if name_like:
        where.append("name LIKE ?")
        args.append(f"%{name_like}%")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY started DESC LIMIT ?"
    args.append(limit)

    out = []
    for row in conn.execute(sql, args):
        run = dict(row)
        run["params"] = {
            r["key"]: json.loads(r["value"])
            for r in conn.execute("SELECT key, value FROM params WHERE run_id=?", (run["id"],))
        }
        run["metric_keys"] = [
            r["key"] for r in conn.execute(
                "SELECT DISTINCT key FROM metrics WHERE run_id=? ORDER BY key", (run["id"],))
        ]
        counts = conn.execute(
            "SELECT COUNT(*) n, MAX(step) s FROM metrics WHERE run_id=?", (run["id"],)
        ).fetchone()
        run["metric_count"] = counts["n"]
        run["last_step"] = counts["s"]
        run["duration"] = (run["ended"] or time.time()) - run["started"]
        out.append(run)
    conn.close()
    return out


def get_metrics(run_id, key=None, db_path=DEFAULT_DB):
    """Return {key: [(step, value, wall_time), ...]} in step order."""
    conn = connect(db_path)
    sql = "SELECT key, step, value, wall_time FROM metrics WHERE run_id=?"
    args = [run_id]
    if key:
        sql += " AND key=?"
        args.append(key)
    sql += " ORDER BY key, step"

    out = {}
    for row in conn.execute(sql, args):
        out.setdefault(row["key"], []).append(
            (row["step"], _decode(row["value"]), row["wall_time"])
        )
    conn.close()
    return out


def get_artifacts(run_id, db_path=DEFAULT_DB):
    conn = connect(db_path)
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM artifacts WHERE run_id=? ORDER BY name", (run_id,))]
    conn.close()
    return rows


def get_run(run_id, db_path=DEFAULT_DB):
    for run in list_runs(db_path, limit=10_000):
        if run["id"] == run_id or run["id"].startswith(run_id):
            return run
    return None


def delete_run(run_id, db_path=DEFAULT_DB, remove_artifacts=True):
    """Remove a run and everything under it. Explicit, never automatic."""
    conn = connect(db_path)
    for table in ("metrics", "params", "artifacts"):
        conn.execute(f"DELETE FROM {table} WHERE run_id=?", (run_id,))
    conn.execute("DELETE FROM runs WHERE id=?", (run_id,))
    conn.commit()
    conn.close()
    directory = os.path.join(ARTIFACT_DIR, run_id)
    if remove_artifacts and os.path.isdir(directory):
        shutil.rmtree(directory)


@contextmanager
def sklearn_run(name, estimator, db_path=DEFAULT_DB, **kwargs):
    """Log an sklearn estimator's hyperparameters automatically.

    Deliberately thin. `estimator.get_params()` is already the whole story for
    sklearn, so wrapping it in a framework would add surface without adding
    information. There is no PyTorch equivalent here for the same reason --
    see README.md, "what is not here".
    """
    params = {k: v for k, v in estimator.get_params().items() if v is not None}
    run = Run(name, params=params, db_path=db_path, **kwargs)
    try:
        yield run
    except Exception as exc:
        run.finish("crashed", f"{type(exc).__name__}: {exc}")
        raise
    else:
        run.finish("finished")
