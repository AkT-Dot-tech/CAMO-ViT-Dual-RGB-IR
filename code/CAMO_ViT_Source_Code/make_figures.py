"""
Paper figures from seed-0 runs:
  figures/qualitative.png   RGB | IR | GT | RGB-only | IR-only | Early | Ours  (8 images)
  figures/curves.png        validation Sα vs epoch for every seed-0 run
python make_figures.py --data_root "D:/CAMO-ViT/VIAC"
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
from PIL import Image  # noqa

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camo_rgbt.common import list_pairs  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--data_root", required=True)
ap.add_argument("--n", type=int, default=8)
a = ap.parse_args()
out = Path("figures")
out.mkdir(exist_ok=True)

cols = [("RGB-only", "rgb_none_b0_ft_s0"), ("IR-only", "ir_none_b0_ft_s0"),
        ("Early fusion", "early_none_b0_ft_s0"), ("Ours (GCMF)", "dual_gated_b0_ft_s0")]
cols = [(t, Path("runs") / r) for t, r in cols if (Path("runs") / r / "preds").exists()]
ours = Path("runs") / "dual_gated_b0_ft_s0"
base = Path("runs") / "rgb_none_b0_ft_s0"

# choose images where ours improves most over RGB-only (+ a couple of small objects)
def load(p):
    return {r["image"]: r for r in csv.DictReader(open(p / "per_image.csv"))}

pairs = {p["rgb"].name: p for p in list_pairs(a.data_root, "test", need_ir=True)}
if (ours / "per_image.csv").exists() and (base / "per_image.csv").exists():
    A, B = load(ours), load(base)
    ks = sorted(set(A) & set(B), key=lambda k: float(B[k]["Sa"]) - float(A[k]["Sa"]))
    pick = ks[: a.n]
else:
    pick = sorted(pairs)[: a.n]

fig, ax = plt.subplots(len(pick), 3 + len(cols), figsize=(2.6 * (3 + len(cols)), 2.3 * len(pick)))
ax = np.atleast_2d(ax)
for r, k in enumerate(pick):
    p = pairs[k]
    rgb = Image.open(p["rgb"]).convert("RGB")
    ims = [(rgb, "RGB"), (Image.open(p["ir"]).convert("L").resize(rgb.size), "IR"),
           (Image.open(p["gt"]).convert("L").resize(rgb.size), "GT")]
    for t, run in cols:
        pr = run / "preds" / (p["gt"].stem + ".png")
        ims.append((Image.open(pr).resize(rgb.size) if pr.exists() else Image.new("L", rgb.size), t))
    for j, (im, t) in enumerate(ims):
        ax[r, j].imshow(im, cmap=None if j == 0 else "gray")
        ax[r, j].axis("off")
        if r == 0:
            ax[r, j].set_title(t, fontsize=11)
plt.tight_layout()
plt.savefig(out / "qualitative.png", dpi=200)
plt.close()

plt.figure(figsize=(7, 4.5))
for h in sorted(Path("runs").glob("*_s0/history.csv")):
    rows = list(csv.DictReader(open(h)))
    plt.plot([int(x["epoch"]) for x in rows], [float(x["val_Sa"]) for x in rows], label=h.parent.name)
plt.xlabel("epoch")
plt.ylabel("val Sα")
plt.legend(fontsize=7)
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig(out / "curves.png", dpi=200)
print("Saved -> figures/")
