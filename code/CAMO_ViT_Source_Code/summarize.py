"""
Collect all runs/*/metrics.json -> paper tables (mean ± std over seeds) + significance test.
python summarize.py
Output: summary/summary_table.csv, summary/summary.md, summary/all_runs.csv
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

runs = sorted(Path("runs").glob("*/metrics.json"))
out = Path("summary")
out.mkdir(exist_ok=True)
allr = [json.load(open(r)) for r in runs if "smoke" not in str(r)]
if not allr:
    raise SystemExit("no runs found")

keys = ["Sa", "Em_mean", "wF", "MAE", "Dice", "IoU"]
with open(out / "all_runs.csv", "w", newline="") as f:
    cols = ["run", "mode", "fusion", "backbone", "frozen", "seed", "best_epoch"] + keys + \
           ["zero_dice_images", "params_M", "gflops", "fps_fp32_bs1", "train_minutes"]
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(allr)

grp = defaultdict(list)
for r in allr:
    grp[r["run"].rsplit("_s", 1)[0]].append(r)

md = ["| Config | #seeds | Sα ↑ | Eφ(mean) ↑ | Fβw ↑ | MAE ↓ | Dice ↑ | IoU ↑ | Small-obj Sα ↑ | Zero-Dice imgs ↓ | Params(M) | GFLOPs | FPS |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
rows = []
for cfg, rs in sorted(grp.items()):
    def ms(k, src=rs):
        v = np.array([x[k] for x in src], dtype=float)
        return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)
    small = np.array([x["by_size"]["small(<1%)"]["Sa"] for x in rs], dtype=float)
    row = {"config": cfg, "n_seeds": len(rs)}
    cells = []
    for k in keys:
        m, s = ms(k)
        row[k], row[k + "_std"] = m, s
        cells.append(f"{m:.4f} ± {s:.4f}")
    zd = ms("zero_dice_images")[0]
    row["small_Sa"] = float(np.nanmean(small))
    row["zero_dice"] = zd
    g = rs[0].get("gflops")
    md.append(f"| {cfg} | {len(rs)} | " + " | ".join(cells) +
              f" | {np.nanmean(small):.4f} | {zd:.1f} | {rs[0]['params_M']:.2f} | "
              f"{g if g is None else round(g, 2)} | {np.mean([x['fps_fp32_bs1'] for x in rs]):.1f} |")
    rows.append(row)

with open(out / "summary_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

# ---------- paired significance: proposed vs each baseline (seed 0, per-image Sα)
md.append("\n### Paired Wilcoxon test on per-image Sα (seed 0)\n")
try:
    from scipy.stats import wilcoxon
    ref = next((p for p in Path("runs").glob("dual_gated_*_ft_s0")), None)
    if ref:
        a = {r["image"]: float(r["Sa"]) for r in csv.DictReader(open(ref / "per_image.csv"))}
        for other in sorted(Path("runs").glob("*_s0")):
            if other == ref or not (other / "per_image.csv").exists():
                continue
            b = {r["image"]: float(r["Sa"]) for r in csv.DictReader(open(other / "per_image.csv"))}
            ks = sorted(set(a) & set(b))
            d = np.array([a[k] - b[k] for k in ks])
            pval = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            md.append(f"- {ref.name} vs {other.name}: mean ΔSα = {d.mean():+.4f}, "
                      f"better on {np.sum(d>0)}/{len(d)} images, p = {pval:.2e}")
except Exception as e:
    md.append(f"(skipped: {e})")

# ---------- size buckets
md.append("\n### By object size (mean over seeds, Sα / Dice)\n")
md.append("| Config | small(<1%) | medium(1-5%) | large(>=5%) | multi-object |")
md.append("|---|---|---|---|---|")
for cfg, rs in sorted(grp.items()):
    cells = []
    for bkt in ["small(<1%)", "medium(1-5%)", "large(>=5%)", "multi-object"]:
        sa = np.nanmean([x["by_size"][bkt]["Sa"] for x in rs])
        dc = np.nanmean([x["by_size"][bkt]["Dice"] for x in rs])
        n = rs[0]["by_size"][bkt]["n"]
        cells.append(f"{sa:.3f} / {dc:.3f} (n={n})")
    md.append(f"| {cfg} | " + " | ".join(cells) + " |")

(out / "summary.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
print("\nSaved -> summary/")
