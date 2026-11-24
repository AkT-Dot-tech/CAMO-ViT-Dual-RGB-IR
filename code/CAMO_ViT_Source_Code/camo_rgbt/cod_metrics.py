"""
Standard Camouflaged Object Detection (COD) metrics.
Re-implementation following PySODMetrics (lartpang/PySODMetrics), which is the
toolbox used by SINet-V2, ZoomNet, FEDER etc. Results are comparable with papers.

Metrics:  S-measure (Sa), mean/max/adaptive E-measure (Em), weighted F-measure (wFb),
          MAE, plus Dice / IoU at threshold 0.5.
Input to every metric: pred = float array in [0,1], gt = bool array (same HxW).
"""
import numpy as np
from scipy.ndimage import convolve, distance_transform_edt

_EPS = np.spacing(1)


def prepare(pred, gt):
    """pred: float [0,1] or uint8; gt: any -> bool. Same normalisation as PySODMetrics."""
    gt = np.asarray(gt)
    if gt.dtype != bool:
        gt = gt > (128 if gt.max() > 1 else 0.5)
    pred = np.asarray(pred, dtype=np.float64)
    if pred.max() > 1.0:
        pred = pred / 255.0
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    return pred, gt


# ---------------------------------------------------------------- MAE
def mae(pred, gt):
    return float(np.mean(np.abs(pred - gt)))


# ---------------------------------------------------------------- S-measure
def _s_object(pred, gt):
    vals = pred[gt]
    if vals.size == 0:
        return 0.0
    x = np.mean(vals)
    sigma_x = np.std(vals, ddof=1) if vals.size > 1 else 0.0
    return 2 * x / (x ** 2 + 1 + sigma_x + _EPS)


def _object(pred, gt):
    fg = pred * gt
    bg = (1 - pred) * (~gt)
    u = np.mean(gt)
    return u * _s_object(fg, gt) + (1 - u) * _s_object(bg, ~gt)


def _centroid(gt):
    h, w = gt.shape
    if gt.sum() == 0:
        x, y = int(round(w / 2)), int(round(h / 2))
    else:
        iy, ix = np.nonzero(gt)
        x, y = int(round(ix.mean())), int(round(iy.mean()))
    return x + 1, y + 1


def _ssim(pred, gt):
    h, w = pred.shape
    n = h * w
    if n <= 1:
        return 0.0
    gt = gt.astype(np.float64)
    x, y = pred.mean(), gt.mean()
    sx = ((pred - x) ** 2).sum() / (n - 1)
    sy = ((gt - y) ** 2).sum() / (n - 1)
    sxy = ((pred - x) * (gt - y)).sum() / (n - 1)
    alpha = 4 * x * y * sxy
    beta = (x ** 2 + y ** 2) * (sx + sy)
    if alpha != 0:
        return alpha / (beta + _EPS)
    if alpha == 0 and beta == 0:
        return 1.0
    return 0.0


def _region(pred, gt):
    h, w = gt.shape
    area = h * w
    x, y = _centroid(gt)
    w1 = x * y / area
    w2 = y * (w - x) / area
    w3 = (h - y) * x / area
    w4 = 1 - w1 - w2 - w3
    parts = [
        (pred[0:y, 0:x], gt[0:y, 0:x], w1),
        (pred[0:y, x:w], gt[0:y, x:w], w2),
        (pred[y:h, 0:x], gt[y:h, 0:x], w3),
        (pred[y:h, x:w], gt[y:h, x:w], w4),
    ]
    return sum(wt * _ssim(p, g) for p, g, wt in parts if p.size > 0)


def s_measure(pred, gt, alpha=0.5):
    y = np.mean(gt)
    if y == 0:
        return float(1 - np.mean(pred))
    if y == 1:
        return float(np.mean(pred))
    sm = alpha * _object(pred, gt) + (1 - alpha) * _region(pred, gt)
    return float(max(0.0, sm))


# ---------------------------------------------------------------- E-measure
def _em_from_counts(fg_fg, fg_bg, gt_fg, gt_size):
    """vectorised enhanced-alignment score for arrays of thresholds."""
    fg_fg = np.asarray(fg_fg, dtype=np.float64)
    fg_bg = np.asarray(fg_bg, dtype=np.float64)
    pred_fg = fg_fg + fg_bg
    pred_bg = gt_size - pred_fg
    if gt_fg == 0:
        s = pred_bg
    elif gt_fg == gt_size:
        s = pred_fg
    else:
        bg_fg = gt_fg - fg_fg
        bg_bg = pred_bg - bg_fg
        mp = pred_fg / gt_size
        mg = gt_fg / gt_size
        combos = [
            (fg_fg, 1 - mp, 1 - mg),
            (fg_bg, 1 - mp, 0 - mg),
            (bg_fg, 0 - mp, 1 - mg),
            (bg_bg, 0 - mp, 0 - mg),
        ]
        s = 0
        for numel, a, b in combos:
            align = 2 * a * b / (a ** 2 + b ** 2 + _EPS)
            s = s + ((align + 1) ** 2 / 4) * numel
    return s / (gt_size - 1 + _EPS)


def e_measure(pred, gt):
    """returns dict(mean=, max=, adp=) like PySODMetrics."""
    gt_fg = int(gt.sum())
    gt_size = gt.size
    # adaptive
    thr = min(2 * pred.mean(), 1.0)
    b = pred >= thr
    adp = float(_em_from_counts(np.count_nonzero(b & gt), np.count_nonzero(b & ~gt), gt_fg, gt_size))
    # curve over 256 thresholds
    p8 = (pred * 255).astype(np.uint8)
    bins = np.linspace(0, 256, 257)
    h_fg, _ = np.histogram(p8[gt], bins=bins)
    h_bg, _ = np.histogram(p8[~gt], bins=bins)
    fg_fg = np.cumsum(np.flip(h_fg))
    fg_bg = np.cumsum(np.flip(h_bg))
    curve = _em_from_counts(fg_fg, fg_bg, gt_fg, gt_size)
    return {"mean": float(np.mean(curve)), "max": float(np.max(curve)), "adp": adp}


# ---------------------------------------------------------------- weighted F
def _gauss2d(shape=(7, 7), sigma=5):
    m, n = [(ss - 1) / 2 for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    s = h.sum()
    return h / s if s != 0 else h


_K = _gauss2d()


def weighted_f(pred, gt, beta=1):
    if not gt.any():
        return 0.0
    dst, idxt = distance_transform_edt(gt == 0, return_indices=True)
    E = np.abs(pred - gt)
    Et = np.copy(E)
    Et[gt == 0] = Et[idxt[0][gt == 0], idxt[1][gt == 0]]
    EA = convolve(Et, weights=_K, mode="constant", cval=0)
    MIN_E_EA = np.where(gt & (EA < E), EA, E)
    B = np.where(gt == 0, 2 - np.exp(np.log(0.5) / 5 * dst), np.ones_like(gt, dtype=np.float64))
    Ew = MIN_E_EA * B
    TPw = np.sum(gt) - np.sum(Ew[gt == 1])
    FPw = np.sum(Ew[gt == 0])
    R = 1 - np.mean(Ew[gt == 1])
    P = TPw / (TPw + FPw + _EPS)
    return float((1 + beta) * R * P / (R + beta * P + _EPS))


# ---------------------------------------------------------------- Dice / IoU
def dice_iou(pred, gt, thr=0.5):
    b = pred >= thr
    tp = np.count_nonzero(b & gt)
    fp = np.count_nonzero(b & ~gt)
    fn = np.count_nonzero(~b & gt)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    return float(dice), float(iou), tp, fp, fn


def all_metrics(pred, gt, full=True):
    """pred float[0,1], gt bool. full=False -> fast subset for validation."""
    raw = np.asarray(pred, dtype=np.float64)
    raw = raw / 255.0 if raw.max() > 1.0 else raw
    pred, gt = prepare(pred, gt)
    d, i, tp, fp, fn = dice_iou(raw, gt)          # Dice/IoU on raw probability @0.5
    tp, fp, fn = int(tp), int(fp), int(fn)
    out = {"Sa": s_measure(pred, gt), "MAE": mae(pred, gt), "Dice": d, "IoU": i,
           "tp": tp, "fp": fp, "fn": fn}
    if full:
        em = e_measure(pred, gt)
        out.update({"Em_mean": em["mean"], "Em_max": em["max"], "Em_adp": em["adp"],
                    "wF": weighted_f(pred, gt)})
    return out
