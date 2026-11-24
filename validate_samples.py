
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from inference import CAMOViTInference

# --------------------------------------------------
# PATHS
# --------------------------------------------------
DATASET = Path(
    r"C:\Users\aksha\Downloads\Major Project\New folder\VIAC\VIAC\testdata"
)

VISIBLE = DATASET / "Visible"
INFRARED = DATASET / "Infrared"
GT = DATASET / "GT"

CHECKPOINT = Path("model/dual_concat_b0_ft_s1/dual_concat_b0_ft_s1/best.pth")

OUTPUT = Path("validation_results")
OUTPUT.mkdir(exist_ok=True)

# --------------------------------------------------
# INDEX FILES BY FILENAME STEM
# --------------------------------------------------
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def index_images(folder):
    return {
        p.stem: p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    }

rgb_files = index_images(VISIBLE)
ir_files = index_images(INFRARED)
gt_files = index_images(GT)

common = sorted(
    set(rgb_files) & set(ir_files) & set(gt_files),
    key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else s.lower())
)

print(f"Matched triplets: {len(common)}")

if len(common) < 5:
    raise RuntimeError("Fewer than five complete RGB-IR-GT triplets found.")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
print("\nLoading CAMO-ViT...")
model = CAMOViTInference(str(CHECKPOINT))
print(f"Model loaded on: {model.device}")

# --------------------------------------------------
# METRICS
# --------------------------------------------------
def calculate_metrics(prediction, target):
    pred = prediction > 127
    gt = target > 127

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    dice = (2 * intersection) / (pred.sum() + gt.sum() + 1e-8)
    iou = intersection / (union + 1e-8)

    return dice, iou

# --------------------------------------------------
# VALIDATE FIVE SAMPLES
# --------------------------------------------------
results = []

for index, stem in enumerate(common[:5], start=1):
    print(f"\n[{index}/5] Testing {stem}")

    rgb = ImageOps.exif_transpose(Image.open(rgb_files[stem])).convert("RGB")
    ir = ImageOps.exif_transpose(Image.open(ir_files[stem])).convert("L")
    gt = ImageOps.exif_transpose(Image.open(gt_files[stem])).convert("L")

    # Check source image alignment dimensions
    rgb_ratio = rgb.width / rgb.height
    ir_ratio = ir.width / ir.height

    if abs(rgb_ratio - ir_ratio) / rgb_ratio > 0.01:
        print("WARNING: RGB and IR aspect ratios differ. Check alignment.")

    # Predict
    prediction = model.predict(rgb, ir)
    pred_mask = prediction["mask"]

    # Resize GT using nearest-neighbor interpolation
    gt_resized = gt.resize(
        (pred_mask.shape[1], pred_mask.shape[0]),
        Image.Resampling.NEAREST
    )
    gt_array = np.array(gt_resized)

    dice, iou = calculate_metrics(pred_mask, gt_array)

    results.append((dice, iou, prediction["inference_time"]))

    print(f"Dice: {dice:.4f}")
    print(f"IoU:  {iou:.4f}")
    print(f"Inference: {prediction['inference_time']:.4f} sec")

    # Save prediction
    pred_image = Image.fromarray(pred_mask)
    pred_image.save(OUTPUT / f"{stem}_prediction.png")

    # Save ground truth
    gt_resized.save(OUTPUT / f"{stem}_ground_truth.png")

    # Save visual comparison
    rgb_preview = rgb.resize((384, 384))
    gt_preview = gt_resized.resize((384, 384))
    pred_preview = pred_image.resize((384, 384))

    comparison = Image.new("RGB", (1152, 420), "white")
    comparison.paste(rgb_preview, (0, 30))
    comparison.paste(gt_preview.convert("RGB"), (384, 30))
    comparison.paste(pred_preview.convert("RGB"), (768, 30))

    draw = ImageDraw.Draw(comparison)
    draw.text((10, 8), "RGB", fill="black")
    draw.text((394, 8), "Ground Truth", fill="black")
    draw.text((778, 8), "CAMO-ViT Prediction", fill="black")

    comparison.save(OUTPUT / f"{stem}_comparison.png")

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------
print("\n" + "=" * 45)
print("FIVE-SAMPLE VALIDATION SUMMARY")
print("=" * 45)

print(f"Mean Dice: {np.mean([r[0] for r in results]):.4f}")
print(f"Mean IoU:  {np.mean([r[1] for r in results]):.4f}")
print(f"Mean inference time: {np.mean([r[2] for r in results]):.4f} sec")

print(f"\nSaved comparison images to: {OUTPUT.resolve()}")
