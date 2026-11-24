"""
Train + validate + test one configuration on VIAC and save everything needed for the paper.

Example:
  python train_eval.py --data_root "D:/CAMO-ViT/VIAC" --mode dual --fusion gated --seed 0
  python train_eval.py --data_root "D:/CAMO-ViT/VIAC" --mode rgb --seed 0
  python train_eval.py --data_root ... --mode rgb --freeze_encoder --seed 0   (old project setting)
Smoke test (2 min):
  python train_eval.py --data_root ... --mode dual --epochs 1 --limit 16 --run_name smoke
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import label as cc_label
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camo_rgbt.common import (CamoNet, VIACRGBT, list_pairs, set_seed, structure_loss,  # noqa
                              train_val_split)
from camo_rgbt.cod_metrics import all_metrics  # noqa

p = argparse.ArgumentParser()
p.add_argument("--data_root", required=True, help="folder that contains Traindata/ and Testdata/")
p.add_argument("--mode", default="dual", choices=["rgb", "ir", "early", "dual"])
p.add_argument("--fusion", default="gated", choices=["gated", "add", "concat"])
p.add_argument("--backbone", default="b0", choices=["b0", "b1", "b2"])
p.add_argument("--freeze_encoder", action="store_true")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--patience", type=int, default=10)
p.add_argument("--batch_size", type=int, default=4)
p.add_argument("--image_size", type=int, default=384)
p.add_argument("--lr", type=float, default=6e-5, help="encoder LR; decoder/fusion get 10x")
p.add_argument("--workers", type=int, default=0)
p.add_argument("--limit", type=int, default=0, help="use only N images per split (smoke test)")
p.add_argument("--save_preds", type=int, default=-1, help="1/0; default: only for seed 0")
p.add_argument("--out_root", default="runs")
p.add_argument("--run_name", default="")
p.add_argument("--fresh", action="store_true")
args = p.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(args.seed)
torch.backends.cudnn.benchmark = True

fusion_tag = args.fusion if args.mode == "dual" else "none"
name = args.run_name or (f"{args.mode}_{fusion_tag}_{args.backbone}_"
                         f"{'frozen' if args.freeze_encoder else 'ft'}_s{args.seed}")
out = Path(args.out_root) / name
out.mkdir(parents=True, exist_ok=True)
save_preds = (args.seed == 0) if args.save_preds < 0 else bool(args.save_preds)
log_f = open(out / "log.txt", "a", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log_f.write(s + "\n")
    log_f.flush()


log("=" * 70)
log("RUN:", name, "| device:", dev,
    "| gpu:", torch.cuda.get_device_name(0) if dev.type == "cuda" else "-")
log("args:", vars(args))

# ---------------------------------------------------------------
# SAFETY: validate existing experiment before resuming or restarting
config_path = out / "config.json"
last_path = out / "last.pth"
best_path = out / "best.pth"
tmp_path = out / "last.tmp"

critical_keys = [
    "data_root", "mode", "fusion", "backbone",
    "freeze_encoder", "seed", "epochs", "patience",
    "batch_size", "image_size", "lr", "workers", "limit",
]

if args.fresh and any(p.exists() for p in
                      [last_path, best_path, tmp_path]):
    raise RuntimeError(
        "FRESH RUN REFUSED: checkpoint files already exist. "
        "Use a new run_name or safely archive the old run first."
    )

if last_path.exists():
    if not config_path.exists():
        raise RuntimeError(
            "Cannot safely resume: last.pth exists but config.json is missing."
        )

    old_config = json.loads(config_path.read_text())
    current_config = vars(args)

    mismatches = [
        key for key in critical_keys
        if old_config.get(key) != current_config.get(key)
    ]

    if mismatches:
        raise RuntimeError(
            "CHECKPOINT CONFIGURATION MISMATCH: "
            + ", ".join(mismatches)
            + ". Refusing to resume."
        )

    log("CHECKPOINT CONFIGURATION VERIFIED.")

elif best_path.exists() or tmp_path.exists():
    raise RuntimeError(
        "Orphan checkpoint detected (best.pth or last.tmp). "
        "Inspect or archive this run before continuing."
    )

else:
    config_path.write_text(
        json.dumps(vars(args), indent=2)
    )
    log("NEW EXPERIMENT CONFIGURATION SAVED.")


# ------------------------------------------------------------------ data
need_ir = args.mode != "rgb"
train_all = list_pairs(args.data_root, "train", need_ir=need_ir)
test_pairs = list_pairs(args.data_root, "test", need_ir=need_ir)
tr_pairs, va_pairs = train_val_split(train_all, 0.15, seed=42)   # fixed split for all runs
if args.limit:
    tr_pairs, va_pairs, test_pairs = tr_pairs[:args.limit], va_pairs[:args.limit], test_pairs[:args.limit]
log(f"train {len(tr_pairs)} | val {len(va_pairs)} | test {len(test_pairs)}")

ds_tr = VIACRGBT(tr_pairs, args.image_size, train=True, use_ir=need_ir)
ds_va = VIACRGBT(va_pairs, args.image_size, train=False, use_ir=need_ir)
ds_te = VIACRGBT(test_pairs, args.image_size, train=False, use_ir=need_ir)
g = torch.Generator()
g.manual_seed(args.seed)
dl_tr = DataLoader(ds_tr, args.batch_size, shuffle=True, num_workers=args.workers,
                   drop_last=True, pin_memory=dev.type == "cuda", generator=g)
dl_va = DataLoader(ds_va, 1, shuffle=False, num_workers=args.workers)
dl_te = DataLoader(ds_te, 1, shuffle=False, num_workers=args.workers)

# ------------------------------------------------------------------ model
model = CamoNet(args.mode, args.fusion, args.backbone).to(dev)
enc_ids = {id(q) for q in model.encoder_params()}
if args.freeze_encoder:
    for q in model.encoder_params():
        q.requires_grad = False
enc_p = [q for q in model.parameters() if id(q) in enc_ids and q.requires_grad]
oth_p = [q for q in model.parameters() if id(q) not in enc_ids and q.requires_grad]
groups = [{"params": oth_p, "lr": args.lr * 10}]
if enc_p:
    groups.append({"params": enc_p, "lr": args.lr})
opt = torch.optim.AdamW(groups, weight_decay=0.01)
total_iters = args.epochs * max(1, len(dl_tr))
warm = max(1, len(dl_tr))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda it: (it + 1) / warm if it < warm else max(0.0, 1 - (it - warm) / max(1, total_iters - warm)) ** 0.9)
scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

n_params = sum(q.numel() for q in model.parameters())
n_train = sum(q.numel() for q in model.parameters() if q.requires_grad)
log(f"params total {n_params/1e6:.2f}M | trainable {n_train/1e6:.2f}M")


def set_train_mode():
    model.train()
    if args.freeze_encoder:
        model.enc.eval()
        if args.mode == "dual":
            model.enc_ir.eval()


@torch.no_grad()
def predict(batch):
    with torch.amp.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
        lg = model(batch["rgb"].to(dev), batch["ir"].to(dev))
    return torch.sigmoid(lg.float())


# ------------------------------------------------------------------ train
best, bad, hist = -1, 0, []
ckpt = out / "best.pth"
last_ckpt = out / "last.pth"
start_ep = 1
elapsed_before = 0.0

if last_ckpt.exists() and not args.fresh:
    state = torch.load(last_ckpt, map_location="cpu", weights_only=False)

    required_keys = {
        "epoch", "model", "optimizer", "scheduler", "scaler",
        "best", "bad", "history", "elapsed_seconds",
        "loader_generator", "python_rng", "numpy_rng",
        "torch_rng", "cuda_rng"
    }

    missing = required_keys - set(state.keys())
    if missing:
        raise RuntimeError(
            f"Invalid recovery checkpoint. Missing keys: {sorted(missing)}"
        )

    if not isinstance(state["epoch"], int) or state["epoch"] < 1:
        raise RuntimeError("Invalid recovery checkpoint epoch.")

    if not isinstance(state["history"], list):
        raise RuntimeError("Invalid recovery checkpoint history.")

    if len(state["history"]) != state["epoch"]:
        raise RuntimeError(
            "Checkpoint history length does not match saved epoch."
        )

    model.load_state_dict(state["model"])
    opt.load_state_dict(state["optimizer"])
    sched.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    best, bad = state["best"], state["bad"]
    hist = state["history"]
    start_ep = state["epoch"] + 1
    elapsed_before = state["elapsed_seconds"]
    g.set_state(state["loader_generator"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if dev.type == "cuda" and state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    log(f"RECOVERY: resumed at epoch {start_ep}")
elif args.fresh and last_ckpt.exists():
    log("FRESH RUN: ignoring previous checkpoint")

t0 = time.time()
for ep in range(start_ep, args.epochs + 1):
    if bad >= args.patience:
        break
    set_train_mode()
    tl, nb = 0.0, 0
    for b in dl_tr:
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            lg = model(b["rgb"].to(dev), b["ir"].to(dev))
        loss = structure_loss(lg, b["mask"].to(dev))
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        tl += loss.item()
        nb += 1
    model.eval()
    ms = []
    for b in dl_va:
        pr = predict(b)[0, 0].cpu().numpy()
        ms.append(all_metrics(pr, b["mask"][0, 0].numpy() > 0.5, full=False))
    va = {k: float(np.mean([m[k] for m in ms])) for k in ("Sa", "MAE", "Dice", "IoU")}
    row = {"epoch": ep, "train_loss": tl / max(1, nb), **{f"val_{k}": v for k, v in va.items()},
           "minutes": (elapsed_before + time.time() - t0) / 60}
    hist.append(row)
    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        w.writeheader()
        w.writerows(hist)
    improved = va["Sa"] > best
    if improved:
        best, bad = va["Sa"], 0
        torch.save({"model": model.state_dict(), "epoch": ep, "val_Sa": best}, ckpt)
    else:
        bad += 1

    recovery = {
        "epoch": ep,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "scaler": scaler.state_dict(),
        "best": best,
        "bad": bad,
        "history": hist,
        "elapsed_seconds": elapsed_before + time.time() - t0,
        "loader_generator": g.get_state(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if dev.type == "cuda" else None,
    }
    tmp_ckpt = out / "last.tmp"
    torch.save(recovery, tmp_ckpt)
    tmp_ckpt.replace(last_ckpt)
    log(f"RECOVERY CHECKPOINT SAVED: epoch {ep}")

    log(f"ep {ep:3d} | loss {row['train_loss']:.4f} | val Sa {va['Sa']:.4f} Dice {va['Dice']:.4f} "
        f"MAE {va['MAE']:.4f} {'*best*' if improved else ''} | {row['minutes']:.1f} min")
    if bad >= args.patience:
        log("early stopping")
        break

# ------------------------------------------------------------------ test
state = torch.load(ckpt, map_location=dev, weights_only=False)
model.load_state_dict(state["model"])
model.eval()
log(f"\nTEST with best epoch {state['epoch']} (val Sa {state['val_Sa']:.4f})")
pred_dir = out / "preds"
if save_preds:
    pred_dir.mkdir(exist_ok=True)
rows, TP, FP, FN = [], 0, 0, 0
for b in dl_te:
    pinfo = test_pairs[int(b["idx"][0])]
    gt = np.asarray(Image.open(pinfo["gt"]).convert("L")) > 127
    pr = predict(b)
    pr = F.interpolate(pr, size=gt.shape, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    m = all_metrics(pr, gt, full=True)
    TP, FP, FN = TP + m["tp"], FP + m["fp"], FN + m["fn"]
    rows.append({"image": pinfo["rgb"].name, "fg_ratio": float(gt.mean()),
                 "n_objects": int(cc_label(gt)[1]),
                 **{k: m[k] for k in ("Sa", "Em_mean", "Em_max", "Em_adp", "wF", "MAE", "Dice", "IoU")}})
    if save_preds:
        Image.fromarray((pr * 255).astype(np.uint8)).save(pred_dir / (pinfo["gt"].stem + ".png"))

with open(out / "per_image.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)


def mean(k, rs=rows):
    return float(np.mean([r[k] for r in rs])) if rs else float("nan")


res = {k: mean(k) for k in ("Sa", "Em_mean", "Em_max", "Em_adp", "wF", "MAE", "Dice", "IoU")}
res["global_Dice"] = float(2 * TP / (2 * TP + FP + FN + 1e-7))
res["global_IoU"] = float(TP / (TP + FP + FN + 1e-7))
# size buckets (fraction of image covered by object)
buckets = {"small(<1%)": lambda r: r["fg_ratio"] < 0.01,
           "medium(1-5%)": lambda r: 0.01 <= r["fg_ratio"] < 0.05,
           "large(>=5%)": lambda r: r["fg_ratio"] >= 0.05,
           "multi-object": lambda r: r["n_objects"] > 1}
res["by_size"] = {bn: {"n": len([r for r in rows if fn(r)]),
                       "Sa": mean("Sa", [r for r in rows if fn(r)]),
                       "wF": mean("wF", [r for r in rows if fn(r)]),
                       "Dice": mean("Dice", [r for r in rows if fn(r)])}
                  for bn, fn in buckets.items()}
res["zero_dice_images"] = int(sum(r["Dice"] == 0 for r in rows))

# ------------------------------------------------------------------ efficiency
x = torch.randn(1, 3, args.image_size, args.image_size, device=dev)
with torch.no_grad():
    for _ in range(10):
        model(x, x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(50):
        model(x, x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    fps = 50 / (time.time() - t)
gflops = None
try:
    from torch.utils.flop_counter import FlopCounterMode
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x, x)
    gflops = fc.get_total_flops() / 2e9   # MACs-style GFLOPs as in most papers
except Exception as e:  # noqa
    log("flop count skipped:", e)

res.update({"run": name, "mode": args.mode, "fusion": fusion_tag, "backbone": args.backbone,
            "frozen": args.freeze_encoder, "seed": args.seed, "best_epoch": state["epoch"],
            "params_M": n_params / 1e6, "trainable_M": n_train / 1e6, "fps_fp32_bs1": fps,
            "gflops": gflops, "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
            "train_minutes": hist[-1]["minutes"], "n_test": len(rows)})
json.dump(res, open(out / "metrics.json", "w"), indent=2)
log(json.dumps(res, indent=2))
log("DONE ->", out)
