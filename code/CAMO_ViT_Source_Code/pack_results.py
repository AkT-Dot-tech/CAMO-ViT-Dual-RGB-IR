import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEEP_DIRS = ["check_data", "runs", "summary", "figures"]
OUTPUT = ROOT / "results_to_send.zip"

files = []

for folder in KEEP_DIRS:
    base = ROOT / folder

    if base.exists():
        files.extend(
            p for p in base.rglob("*")
            if p.is_file()
            and p.name != "last.tmp"
        )

if not files:
    raise RuntimeError("No result files found. Nothing to back up.")

# Include all files, including .pth model checkpoints.
with zipfile.ZipFile(
    OUTPUT, "w", compression=zipfile.ZIP_DEFLATED
) as z:
    for file in files:
        z.write(file, file.relative_to(ROOT))

# Verify archive integrity.
with zipfile.ZipFile(OUTPUT, "r") as z:
    bad_file = z.testzip()
    if bad_file:
        raise RuntimeError(f"ZIP integrity failure: {bad_file}")

    names = z.namelist()
    checkpoints = [
        n for n in names
        if n.endswith((".pth", ".pt"))
    ]

print("BACKUP CREATED:", OUTPUT)
print("Files included:", len(names))
print("Checkpoint files:", len(checkpoints))

for name in checkpoints:
    print("  ", name)

print("ZIP size: {:.2f} MB".format(OUTPUT.stat().st_size / 1e6))

if not checkpoints:
    print("WARNING: No model checkpoints found in this backup.")
else:
    print("CHECKPOINT INCLUSION: VERIFIED")
