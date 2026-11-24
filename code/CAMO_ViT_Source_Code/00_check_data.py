"""
STEP 0: check that the VIAC download has Visible + Infrared + GT for train and test,
that they pair up, and save a sample grid so alignment can be seen.

python 00_check_data.py --data_root "D:/CAMO-ViT/VIAC"
Output: check_data/report.txt , check_data/samples_train.png , check_data/samples_test.png
"""
import argparse
import platform
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camo_rgbt.common import _index, find_split_dirs  # noqa

p = argparse.ArgumentParser()
p.add_argument("--data_root", required=True)
args = p.parse_args()
out = Path("check_data")
out.mkdir(exist_ok=True)
lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


say("python", platform.python_version(), "|", platform.platform())
try:
    import torch
    import transformers
    say("torch", torch.__version__, "| transformers", transformers.__version__,
        "| cuda", torch.cuda.is_available(),
        "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        f"| VRAM {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB" if torch.cuda.is_available() else "")
except Exception as e:
    say("torch/transformers problem:", e)

say("\nTop-level of data_root:", [d.name for d in Path(args.data_root).iterdir()])
for split in ["train", "test"]:
    say("\n" + "=" * 60, "\nSPLIT:", split)
    d = find_split_dirs(args.data_root, split)
    say("split folder :", d["split_dir"])
    say("sub-folders  :", [x.name for x in d["split_dir"].iterdir()])
    for k in ["visible", "infrared", "gt"]:
        say(f"{k:9s}: {d[k]}")
    vis, ir, gt = _index(d["visible"]), _index(d["infrared"]), _index(d["gt"])
    say(f"counts  visible={len(vis)}  infrared={len(ir)}  gt={len(gt)}")
    common = sorted(set(vis) & set(ir) & set(gt))
    say("complete triplets (rgb+ir+gt):", len(common))
    say("visible without IR (first 10):", sorted(set(vis) - set(ir))[:10])
    say("example names:", sorted(vis)[:5], "| IR:", sorted(ir)[:5])
    if not common:
        say("!!! NO INFRARED PAIRS FOUND for", split)
        continue
    size_mismatch = 0
    for k in common:
        with Image.open(vis[k]) as a, Image.open(ir[k]) as b:
            if a.size != b.size:
                size_mismatch += 1
    say("RGB/IR size mismatches:", size_mismatch)
    with Image.open(vis[common[0]]) as a, Image.open(ir[common[0]]) as b, Image.open(gt[common[0]]) as c:
        say("sample sizes rgb/ir/gt:", a.size, b.size, c.size, "| ir mode:", b.mode)
    # foreground ratio stats
    fr = []
    for k in common:
        g = np.asarray(Image.open(gt[k]).convert("L")) > 127
        fr.append(g.mean())
    fr = np.array(fr)
    say(f"object area %: mean {fr.mean()*100:.2f} | small(<1%) {np.sum(fr<0.01)} | "
        f"medium {np.sum((fr>=0.01)&(fr<0.05))} | large(>=5%) {np.sum(fr>=0.05)} | empty {np.sum(fr==0)}")
    # sample grid
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        random.seed(1)
        pick = random.sample(common, min(6, len(common)))
        fig, ax = plt.subplots(len(pick), 4, figsize=(16, 3.6 * len(pick)))
        for r, k in enumerate(pick):
            a = Image.open(vis[k]).convert("RGB")
            b = Image.open(ir[k]).convert("L").resize(a.size)
            c = np.asarray(Image.open(gt[k]).convert("L").resize(a.size)) > 127
            ov = np.asarray(b.convert("RGB")).astype(float)
            ov[c] = 0.5 * ov[c] + 0.5 * np.array([255, 0, 0])
            for j, (im, t) in enumerate([(a, f"RGB {k}"), (b, "IR"), (c, "GT"), (ov.astype(np.uint8), "IR + GT")]):
                ax[r, j].imshow(im, cmap="gray" if j in (1, 2) else None)
                ax[r, j].set_title(t)
                ax[r, j].axis("off")
        plt.tight_layout()
        plt.savefig(out / f"samples_{split}.png", dpi=80)
        plt.close()
    except Exception as e:
        say("plot skipped:", e)

(out / "report.txt").write_text("\n".join(lines), encoding="utf-8")
print("\nSaved -> check_data/report.txt and sample images. Send this folder.")
