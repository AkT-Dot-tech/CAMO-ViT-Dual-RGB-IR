"""
Shared code: VIAC dataset (visible + infrared + GT), models, loss.

Modes
  rgb    : visible only (SegFormer)                     -> baseline (current project)
  ir     : infrared only (SegFormer)                    -> baseline
  early  : 4-channel input [R,G,B,IR] (SegFormer)       -> baseline (input-level fusion)
  dual   : two SegFormer encoders + per-stage fusion    -> PROPOSED
           fusion = gated  (proposed Gated Cross-Modal Fusion, GCMF)
                    add    (ablation)
                    concat (ablation)
"""
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

HF_NAMES = {
    "b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "b1": "nvidia/segformer-b1-finetuned-ade-512-512",
    "b2": "nvidia/segformer-b2-finetuned-ade-512-512",
}


# =============================================================== data discovery
def _find_child(folder, keys):
    """case-insensitive search for a sub-folder whose name contains one of keys."""
    if folder is None or not Path(folder).exists():
        return None
    for d in sorted(Path(folder).iterdir()):
        if d.is_dir() and any(k in d.name.lower() for k in keys):
            return d
    return None


def find_split_dirs(data_root, split):
    """returns dict(visible=, infrared=, gt=) for split in {'train','test'}."""
    root = Path(data_root)
    split_dir = _find_child(root, [split])
    if split_dir is None:
        raise FileNotFoundError(f"No '{split}*' folder inside {root}")
    return {
        "split_dir": split_dir,
        "visible": _find_child(split_dir, ["visible", "rgb", "vis"]),
        "infrared": _find_child(split_dir, ["infrared", "thermal", "ir", "tir"]),
        "gt": _find_child(split_dir, ["gt", "mask", "label"]),
    }


def _index(folder):
    if folder is None:
        return {}
    return {p.stem.lower(): p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS}


def list_pairs(data_root, split, need_ir=True):
    d = find_split_dirs(data_root, split)
    vis, ir, gt = _index(d["visible"]), _index(d["infrared"]), _index(d["gt"])
    if not vis or not gt:
        raise FileNotFoundError(f"Visible/GT images not found for split '{split}': {d}")
    if need_ir and not ir:
        raise FileNotFoundError(
            f"Infrared folder not found / empty for split '{split}'. Folders seen: {d}")
    # if IR exists we ALWAYS keep only complete triplets, so every mode uses the same images
    keys = sorted(set(vis) & set(gt) & (set(ir) if ir else set(vis)))
    return [{"id": k, "rgb": vis[k], "ir": ir.get(k), "gt": gt[k]} for k in keys]


def train_val_split(pairs, val_ratio=0.15, seed=42):
    ids = [p["id"] for p in pairs]
    rng = random.Random(seed)
    ids_sorted = sorted(ids)
    rng.shuffle(ids_sorted)
    val_ids = set(ids_sorted[: int(len(ids_sorted) * val_ratio)])
    return [p for p in pairs if p["id"] not in val_ids], [p for p in pairs if p["id"] in val_ids]


# =============================================================== dataset
class VIACRGBT(Dataset):
    def __init__(self, pairs, image_size=384, train=False, use_ir=True):
        self.pairs, self.size, self.train, self.use_ir = pairs, image_size, train, use_ir

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _jitter(img, strength=0.2):
        a = np.asarray(img, dtype=np.float32)
        b = 1 + random.uniform(-strength, strength)   # brightness
        c = 1 + random.uniform(-strength, strength)   # contrast
        m = a.mean()
        a = np.clip((a - m) * c + m * b, 0, 255)
        return Image.fromarray(a.astype(np.uint8))

    def __getitem__(self, i):
        p = self.pairs[i]
        rgb = Image.open(p["rgb"]).convert("RGB")
        W, H = rgb.size
        gt = Image.open(p["gt"]).convert("L")
        if gt.size != (W, H):
            gt = gt.resize((W, H), Image.Resampling.NEAREST)
        if self.use_ir and p["ir"] is not None:
            ir = Image.open(p["ir"]).convert("L")
            if ir.size != (W, H):
                ir = ir.resize((W, H), Image.Resampling.BILINEAR)
        else:
            ir = Image.new("L", (W, H), 0)

        if self.train:
            # same geometric aug for all three
            if random.random() < 0.5:
                rgb, ir, gt = [x.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for x in (rgb, ir, gt)]
            if random.random() < 0.7:
                s = random.uniform(0.75, 1.0)
                cw, ch = int(W * s), int(H * s)
                x0, y0 = random.randint(0, W - cw), random.randint(0, H - ch)
                box = (x0, y0, x0 + cw, y0 + ch)
                rgb, ir, gt = rgb.crop(box), ir.crop(box), gt.crop(box)
            rgb = self._jitter(rgb)
            if self.use_ir:
                ir = self._jitter(ir)

        S = (self.size, self.size)
        rgb = np.asarray(rgb.resize(S, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        ir = np.asarray(ir.resize(S, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        gt_small = np.asarray(gt.resize(S, Image.Resampling.NEAREST)) > 127

        rgb = (rgb - MEAN) / STD
        ir3 = (np.repeat(ir[..., None], 3, axis=2) - MEAN) / STD
        return {
            "rgb": torch.from_numpy(rgb).permute(2, 0, 1).float(),
            "ir": torch.from_numpy(ir3).permute(2, 0, 1).float(),
            "mask": torch.from_numpy(gt_small.astype(np.float32))[None],
            "idx": i,
        }


# =============================================================== model
class GatedFusion(nn.Module):
    """Proposed Gated Cross-Modal Fusion (GCMF):
    channel re-weighting of each modality + spatial gate that decides, per pixel,
    how much to trust RGB vs IR, + residual projection of the concatenation."""

    def __init__(self, c):
        super().__init__()
        hid = max(c // 4, 8)
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(2 * c, hid, 1),
                                nn.ReLU(inplace=True), nn.Conv2d(hid, 2 * c, 1), nn.Sigmoid())
        self.sa = nn.Sequential(nn.Conv2d(4, 1, 7, padding=3), nn.Sigmoid())
        self.proj = nn.Sequential(nn.Conv2d(2 * c, c, 1, bias=False), nn.BatchNorm2d(c),
                                  nn.ReLU(inplace=True))
        self.last_gate = None

    def forward(self, r, t):
        x = torch.cat([r, t], 1)
        wr, wt = self.ca(x).chunk(2, 1)
        r2, t2 = r * wr, t * wt
        s = self.sa(torch.cat([r2.mean(1, keepdim=True), r2.amax(1, keepdim=True),
                               t2.mean(1, keepdim=True), t2.amax(1, keepdim=True)], 1))
        self.last_gate = s.detach()
        return s * r2 + (1 - s) * t2 + self.proj(x)


class AddFusion(nn.Module):
    def __init__(self, c):
        super().__init__()

    def forward(self, r, t):
        return r + t


class ConcatFusion(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(2 * c, c, 1, bias=False), nn.BatchNorm2d(c),
                                  nn.ReLU(inplace=True))

    def forward(self, r, t):
        return self.proj(torch.cat([r, t], 1))


FUSIONS = {"gated": GatedFusion, "add": AddFusion, "concat": ConcatFusion}


class CamoNet(nn.Module):
    def __init__(self, mode="dual", fusion="gated", backbone="b0"):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation, SegformerModel
        name = HF_NAMES[backbone]
        base = SegformerForSemanticSegmentation.from_pretrained(
            name, num_labels=1, ignore_mismatched_sizes=True)
        self.mode = mode
        self.enc = base.segformer
        self.head = base.decode_head
        hidden = list(base.config.hidden_sizes)

        if mode == "early":
            pe = self.enc.stages[0].patch_embeddings.proj
            new = nn.Conv2d(4, pe.out_channels, pe.kernel_size, pe.stride, pe.padding,
                            bias=pe.bias is not None)
            with torch.no_grad():
                new.weight[:, :3] = pe.weight
                new.weight[:, 3:] = pe.weight.mean(1, keepdim=True)
                if pe.bias is not None:
                    new.bias.copy_(pe.bias)
            self.enc.stages[0].patch_embeddings.proj = new
            self.enc.config.num_channels = 4

        if mode == "dual":
            self.enc_ir = SegformerModel.from_pretrained(name)
            self.fuse = nn.ModuleList([FUSIONS[fusion](c) for c in hidden])

    def encoder_params(self):
        ps = list(self.enc.parameters())
        if self.mode == "dual":
            ps += list(self.enc_ir.parameters())
        return ps

    @staticmethod
    def _feats(enc, x):
        return enc(pixel_values=x, output_hidden_states=True, return_dict=True).hidden_states

    def forward(self, rgb, ir):
        if self.mode == "rgb":
            hs = self._feats(self.enc, rgb)
        elif self.mode == "ir":
            hs = self._feats(self.enc, ir)
        elif self.mode == "early":
            hs = self._feats(self.enc, torch.cat([rgb, ir.mean(1, keepdim=True)], 1))
        else:
            hr = self._feats(self.enc, rgb)
            ht = self._feats(self.enc_ir, ir)
            hs = [f(a, b) for f, a, b in zip(self.fuse, hr, ht)]
        logits = self.head(list(hs))                      # B,1,H/4,W/4
        return F.interpolate(logits, size=rgb.shape[-2:], mode="bilinear", align_corners=False)


# =============================================================== loss
def structure_loss(logits, mask):
    """weighted BCE + weighted IoU (PraNet / SINet-V2 standard COD loss)."""
    logits = logits.float()
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, 31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    wbce = (weit * wbce).sum((2, 3)) / weit.sum((2, 3))
    p = torch.sigmoid(logits)
    inter = ((p * mask) * weit).sum((2, 3))
    union = ((p + mask) * weit).sum((2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
