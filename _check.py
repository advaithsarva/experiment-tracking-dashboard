import tracker, detect
runs = tracker.list_runs('runs.db')
ok = 0; rows = []
for r in runs:
    m = tracker.get_metrics(r['id'], db_path='runs.db')
    tr = [(s, v) for s, v, _ in m.get('train_loss', [])]
    va = [(s, v) for s, v, _ in m.get('val_loss', [])]
    d, frac = detect.diagnose_incrementally(tr, va)
    t = r['params']['shape']
    hit = (d.kind == t)
    ok += hit
    rows.append((t, d.kind, frac, hit))
for t, k, f, h in sorted(rows):
    mark = 'OK' if h else 'MISS'
    print("  %-12s -> %-12s at %5.1f%%  %s" % (t, k, f * 100, mark))
print()
print("%d/%d correct" % (ok, len(runs)))
