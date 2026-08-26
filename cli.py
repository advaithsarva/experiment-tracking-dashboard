"""Command line front end and dashboard generator.

    python cli.py runs                          # list runs
    python cli.py show <run-id>                 # one run in full
    python cli.py compare <id> <id> [<id>...]   # side by side
    python cli.py diagnose                      # health of every run
    python cli.py dashboard board.html          # the whole thing as one page
    python cli.py export runs.csv               # CSV of every run
    python cli.py delete <run-id>

`--json` on any subcommand writes one JSON document to stdout and nothing
else, so other software can drive this.
"""

import argparse
import csv
import html
import json
import sys

import detect
import tracker


def fmt_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def curves_for(run_id, db):
    metrics = tracker.get_metrics(run_id, db_path=db)
    train = [(s, v) for s, v, _ in metrics.get("train_loss", [])]
    val = [(s, v) for s, v, _ in metrics.get("val_loss", [])]
    return train, val, metrics


def cmd_runs(args):
    runs = tracker.list_runs(args.db, limit=args.limit, status=args.status,
                             name_like=args.name)
    if args.json:
        json.dump(runs, sys.stdout, indent=2, default=str)
        print()
        return 0
    if not runs:
        print("no runs found")
        return 0

    print(f"{'id':<14}{'name':<26}{'status':<10}{'steps':>8}{'metrics':>9}"
          f"{'duration':>10}  params")
    for r in runs:
        params = ", ".join(f"{k}={v}" for k, v in list(r["params"].items())[:3])
        print(f"{r['id']:<14}{r['name'][:25]:<26}{r['status']:<10}"
              f"{(r['last_step'] if r['last_step'] is not None else 0):>8}"
              f"{r['metric_count']:>9}{fmt_duration(r['duration']):>10}  {params[:46]}")
    print(f"\n{len(runs)} runs")
    return 0


def cmd_show(args):
    run = tracker.get_run(args.run_id, args.db)
    if not run:
        print(f"no run matching {args.run_id!r}", file=sys.stderr)
        return 1

    train, val, metrics = curves_for(run["id"], args.db)
    diagnosis, fraction = detect.diagnose_incrementally(train, val)
    artifacts = tracker.get_artifacts(run["id"], args.db)

    if args.json:
        json.dump({
            "run": run,
            "metrics": {k: [{"step": s, "value": v, "wall_time": w} for s, v, w in rows]
                        for k, rows in metrics.items()},
            "artifacts": artifacts,
            "diagnosis": {"kind": diagnosis.kind, "step": diagnosis.step,
                          "detail": diagnosis.detail, "confidence": diagnosis.confidence,
                          "suggestion": diagnosis.suggestion,
                          "detectable_at_fraction": round(fraction, 4)},
        }, sys.stdout, indent=2, default=str)
        print()
        return 0

    print(f"run {run['id']}  {run['name']}")
    print(f"  status    {run['status']}" + (f"  ({run['error']})" if run["error"] else ""))
    print(f"  duration  {fmt_duration(run['duration'])}")
    print(f"  metrics   {run['metric_count']:,} values across {len(run['metric_keys'])} keys")

    print("\n  parameters")
    for k, v in run["params"].items():
        print(f"    {k:<18} {v}")

    print("\n  metrics")
    for key, rows in metrics.items():
        values = [v for _, v, _ in rows]
        finite = [v for v in values if v == v and abs(v) != float("inf")]
        best = min(finite) if finite else float("nan")
        print(f"    {key:<18} {len(rows):>6} points   last {values[-1]:<12.6g} best {best:.6g}")

    if artifacts:
        print("\n  artifacts")
        for a in artifacts:
            print(f"    {a['name']:<24}{a['bytes']:>12,} B   sha256 {a['sha256'][:16]}...")

    print(f"\n  diagnosis: {diagnosis.kind}  ({diagnosis.confidence})")
    print(f"    {diagnosis.detail}")
    if diagnosis.is_problem:
        print(f"    detectable at {fraction:.0%} of the run "
              f"-- stopping there would save {1 - fraction:.0%} of its compute")
    if diagnosis.suggestion:
        print(f"    suggestion: {diagnosis.suggestion}")
    return 0


def cmd_compare(args):
    runs = [tracker.get_run(rid, args.db) for rid in args.run_ids]
    missing = [rid for rid, r in zip(args.run_ids, runs) if r is None]
    if missing:
        print(f"no run matching: {', '.join(missing)}", file=sys.stderr)
        return 1

    rows = []
    for run in runs:
        train, val, _ = curves_for(run["id"], args.db)
        diagnosis, fraction = detect.diagnose_incrementally(train, val)
        finite = [v for _, v in train if v == v and abs(v) != float("inf")]
        rows.append({
            "id": run["id"], "name": run["name"], "status": run["status"],
            "params": run["params"],
            "best_train": min(finite) if finite else None,
            "final_train": train[-1][1] if train else None,
            "best_val": min((v for _, v in val if v == v), default=None),
            "steps": run["last_step"], "diagnosis": diagnosis.kind,
        })

    if args.json:
        json.dump(rows, sys.stdout, indent=2, default=str)
        print()
        return 0

    # Only the parameters that actually differ. Printing all twenty when one
    # changed is what makes comparison tables useless.
    all_keys = sorted({k for r in rows for k in r["params"]})
    varying = [k for k in all_keys
               if len({json.dumps(r["params"].get(k), default=str) for r in rows}) > 1]

    print(f"{'id':<14}{'name':<22}{'best train':>12}{'best val':>11}"
          f"{'steps':>7}{'diagnosis':>14}")
    for r in rows:
        bt = f"{r['best_train']:.5g}" if r["best_train"] is not None else "-"
        bv = f"{r['best_val']:.5g}" if r["best_val"] is not None else "-"
        print(f"{r['id']:<14}{r['name'][:21]:<22}{bt:>12}{bv:>11}"
              f"{r['steps'] or 0:>7}{r['diagnosis']:>14}")

    if varying:
        print(f"\nparameters that differ ({len(all_keys) - len(varying)} identical, hidden)")
        print(f"{'id':<14}" + "".join(f"{k[:13]:>15}" for k in varying))
        for r in rows:
            print(f"{r['id']:<14}" + "".join(
                f"{str(r['params'].get(k, '-'))[:13]:>15}" for k in varying))
    else:
        print("\nevery parameter is identical across these runs")

    best = min((r for r in rows if r["best_val"] is not None),
               key=lambda r: r["best_val"], default=None)
    if best:
        print(f"\nbest validation: {best['name']} ({best['id']}) at {best['best_val']:.5g}")
    return 0


def cmd_diagnose(args):
    runs = tracker.list_runs(args.db, limit=args.limit)
    out = []
    for run in runs:
        train, val, _ = curves_for(run["id"], args.db)
        diagnosis, fraction = detect.diagnose_incrementally(train, val)
        out.append((run, diagnosis, fraction))

    if args.json:
        json.dump([{"id": r["id"], "name": r["name"], "kind": d.kind,
                    "step": d.step, "detail": d.detail, "confidence": d.confidence,
                    "suggestion": d.suggestion, "detectable_at": round(f, 4)}
                   for r, d, f in out], sys.stdout, indent=2, default=str)
        print()
        return 0

    problems = [x for x in out if x[1].is_problem]
    print(f"{'id':<14}{'name':<24}{'diagnosis':<14}{'found at':>10}{'saves':>8}  detail")
    for run, d, fraction in out:
        saves = f"{1 - fraction:.0%}" if d.is_problem else "-"
        found = f"{fraction:.0%}" if d.is_problem else "-"
        print(f"{run['id']:<14}{run['name'][:23]:<24}{d.kind:<14}{found:>10}{saves:>8}  "
              f"{d.detail[:60]}")

    print(f"\n{len(problems)} of {len(out)} runs have a problem")
    if problems:
        mean_saved = sum(1 - f for _, _, f in problems) / len(problems)
        print(f"stopping each at detection would have saved {mean_saved:.0%} of "
              f"their compute on average")
    return 0


def cmd_export(args):
    runs = tracker.list_runs(args.db, limit=args.limit)
    keys = sorted({k for r in runs for k in r["params"]})
    with open(args.path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "name", "status", "steps", "duration_s",
                         "best_train", "best_val", "diagnosis"] + keys)
        for run in runs:
            train, val, _ = curves_for(run["id"], args.db)
            diagnosis, _ = detect.diagnose_incrementally(train, val)
            finite = [v for _, v in train if v == v and abs(v) != float("inf")]
            writer.writerow([
                run["id"], run["name"], run["status"], run["last_step"],
                round(run["duration"], 2),
                min(finite) if finite else "",
                min((v for _, v in val if v == v), default=""),
                diagnosis.kind,
            ] + [run["params"].get(k, "") for k in keys])
    print(f"wrote {args.path} ({len(runs)} runs, {len(keys)} parameter columns)")
    return 0


def cmd_delete(args):
    run = tracker.get_run(args.run_id, args.db)
    if not run:
        print(f"no run matching {args.run_id!r}", file=sys.stderr)
        return 1
    tracker.delete_run(run["id"], args.db)
    print(f"deleted {run['id']} ({run['name']}) and its artifacts")
    return 0


# --------------------------------------------------------------------------
# dashboard
# --------------------------------------------------------------------------

def sparkline(points, width=260, height=54):
    """One inline SVG polyline per curve. No plotting library, no PNG files.

    Non-finite values break the line rather than being dropped: a gap is
    honest about where the loss went NaN, whereas skipping the point draws a
    smooth line straight through the most important event in the run.
    """
    finite = [(s, v) for s, v in points if v == v and abs(v) != float("inf")]
    if len(finite) < 2:
        return ""

    steps = [s for s, _ in finite]
    values = [v for _, v in finite]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    x0, x1 = min(steps), max(steps)
    xspan = (x1 - x0) or 1

    coords = " ".join(
        f"{(s - x0) / xspan * width:.1f},{height - (v - lo) / span * height:.1f}"
        for s, v in finite
    )
    broke = len(finite) < len(points)
    marker = (f'<circle cx="{width}" cy="{height / 2}" r="3" fill="#c0563f">'
              f'<title>curve contains non-finite values</title></circle>' if broke else "")
    return (f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'preserveAspectRatio="none"><polyline points="{coords}" fill="none" '
            f'stroke="currentColor" stroke-width="1.5"/>{marker}</svg>')


def cmd_dashboard(args):
    runs = tracker.list_runs(args.db, limit=args.limit)
    if not runs:
        print("no runs to render", file=sys.stderr)
        return 1

    esc = html.escape
    cards = []
    counts = {}

    for run in runs:
        train, val, metrics = curves_for(run["id"], args.db)
        diagnosis, fraction = detect.diagnose_incrementally(train, val)
        counts[diagnosis.kind] = counts.get(diagnosis.kind, 0) + 1

        finite = [v for _, v in train if v == v and abs(v) != float("inf")]
        best = f"{min(finite):.5g}" if finite else "-"
        best_val = min((v for _, v in val if v == v), default=None)
        params = "".join(
            f"<tr><td>{esc(str(k))}</td><td class=n>{esc(str(v))}</td></tr>"
            for k, v in list(run["params"].items())[:8]
        )
        saves = (f'<span class=saves>stopping at detection saves '
                 f'{1 - fraction:.0%}</span>' if diagnosis.is_problem else "")

        cards.append(f"""<article class="card {esc(diagnosis.kind)}">
<header><h3>{esc(run['name'])}</h3><code>{esc(run['id'])}</code></header>
<div class=chart>{sparkline(train)}</div>
<div class=chart>{sparkline(val)}</div>
<p class=verdict><b>{esc(diagnosis.kind)}</b> ({esc(diagnosis.confidence)}) &middot;
{esc(diagnosis.detail)} {saves}</p>
{f'<p class=fix>{esc(diagnosis.suggestion)}</p>' if diagnosis.suggestion else ''}
<table class=params><tr><th>best train</th><td class=n>{best}</td></tr>
<tr><th>best val</th><td class=n>{f'{best_val:.5g}' if best_val is not None else '-'}</td></tr>
<tr><th>steps</th><td class=n>{run['last_step'] or 0}</td></tr>
<tr><th>status</th><td class=n>{esc(run['status'])}</td></tr>{params}</table>
</article>""")

    tiles = "".join(
        f'<div class="tile {esc(k)}"><div class=v>{n}</div><div class=k>{esc(k)}</div></div>'
        for k, n in sorted(counts.items(), key=lambda kv: -kv[1])
    )

    doc = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Experiment dashboard</title><style>
:root{{--bg:#fbfbfa;--fg:#1a1a19;--line:#dcdad5;--muted:#6b6862;
--ok:#4a7a4a;--bad:#c0563f;--warn:#b3843a}}
@media(prefers-color-scheme:dark){{:root{{--bg:#191918;--fg:#eeece7;--line:#38352f;
--muted:#9a968d;--ok:#7fae7f;--bad:#e08b74;--warn:#d4a95f}}}}
body{{margin:0;padding:2rem 1.5rem;background:var(--bg);color:var(--fg);
font:14px/1.55 ui-sans-serif,system-ui,sans-serif}}
main{{max-width:80rem;margin:0 auto}}
h1{{font-size:1.4rem;margin:0 0 .3rem}} .sub{{color:var(--muted);margin:0 0 1.5rem}}
.tiles{{display:flex;gap:.7rem;flex-wrap:wrap;margin-bottom:1.6rem}}
.tile{{border:1px solid var(--line);border-radius:7px;padding:.7rem 1.1rem;min-width:6rem}}
.tile .v{{font-size:1.5rem;font-variant-numeric:tabular-nums}}
.tile .k{{color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}}
.tile.diverged .v,.card.diverged h3{{color:var(--bad)}}
.tile.healthy .v,.card.healthy h3{{color:var(--ok)}}
.tile.overfitting .v,.tile.plateau .v,.tile.unstable .v,
.card.overfitting h3,.card.plateau h3,.card.unstable h3{{color:var(--warn)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(19rem,1fr));gap:1rem}}
.card{{border:1px solid var(--line);border-radius:8px;padding:1rem}}
.card header{{display:flex;justify-content:space-between;align-items:baseline;gap:.5rem}}
.card h3{{font-size:.98rem;margin:0}} .card code{{color:var(--muted);font-size:11px}}
.chart{{color:var(--muted);margin:.4rem 0;overflow:hidden}}
.chart svg{{width:100%;height:54px;display:block}}
.verdict{{font-size:12.5px;margin:.5rem 0 .3rem}}
.fix{{font-size:12px;color:var(--muted);margin:.2rem 0 .5rem;font-style:italic}}
.saves{{color:var(--ok);white-space:nowrap}}
table.params{{width:100%;border-collapse:collapse;font-size:12px;margin-top:.5rem}}
table.params th{{text-align:left;font-weight:500;color:var(--muted);padding:.15rem 0}}
table.params td{{padding:.15rem 0}} .n{{text-align:right;font-variant-numeric:tabular-nums}}
footer{{margin-top:2.5rem;color:var(--muted);font-size:12px}}
</style></head><body><main>
<h1>Experiment dashboard</h1>
<p class=sub>{len(runs)} runs. Upper curve is training loss, lower is validation.
A red dot means the curve contains NaN or infinity -- the gap is where it happened.</p>
<div class=tiles>{tiles}</div>
<div class=grid>{''.join(cards)}</div>
<footer>Generated by experiment-tracking-dashboard from {esc(args.db)}.
Diagnoses come from replaying each curve step by step, so "found at" is when the
problem became detectable, not when it was obvious in hindsight.</footer>
</main></body></html>"""

    with open(args.path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(f"wrote {args.path} ({len(runs)} runs)")
    return 0


def main(argv=None):
    # Shared options live on a parent parser so they are accepted on either
    # side of the subcommand. `cli.py show ID --json` and `cli.py --json show
    # ID` both work; a machine interface that only parses with the flag in one
    # position is a machine interface people give up on.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", default=tracker.DEFAULT_DB)
    common.add_argument("--limit", type=int, default=200)
    common.add_argument("--json", action="store_true",
                        help="write one JSON document to stdout and nothing else")

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], parents=[common])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, **kw):
        return sub.add_parser(name, parents=[common], **kw)

    p = add("runs"); p.add_argument("--status"); p.add_argument("--name")
    p.set_defaults(fn=cmd_runs)
    p = add("show"); p.add_argument("run_id"); p.set_defaults(fn=cmd_show)
    p = add("compare"); p.add_argument("run_ids", nargs="+"); p.set_defaults(fn=cmd_compare)
    p = add("diagnose"); p.set_defaults(fn=cmd_diagnose)
    p = add("dashboard"); p.add_argument("path"); p.set_defaults(fn=cmd_dashboard)
    p = add("export"); p.add_argument("path"); p.set_defaults(fn=cmd_export)
    p = add("delete"); p.add_argument("run_id"); p.set_defaults(fn=cmd_delete)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
