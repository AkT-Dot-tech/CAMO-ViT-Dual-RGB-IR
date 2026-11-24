@echo off
REM ===== PHASE 3 (optional) : bada backbone B2. Sirf tab jab GPU me 8GB+ VRAM ho =====
set DATA=C:\Users\aksha\Downloads\Major Project\CAMO-ViT\VIAC
python train_eval.py --data_root "%DATA%" --batch_size 4 --backbone b2 --mode rgb  --seed 0 || echo b2_rgb >> failed.txt
python train_eval.py --data_root "%DATA%" --batch_size 4 --backbone b2 --mode dual --fusion gated --seed 0 || echo b2_dual >> failed.txt
python summarize.py
python pack_results.py
pause
