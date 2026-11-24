@echo off
REM ===== 5 minute ka test: sab theek chal raha hai ya nahi =====
set DATA=C:\Users\aksha\Downloads\Major Project\CAMO-ViT\VIAC
python train_eval.py --data_root "%DATA%" --mode dual --fusion gated --epochs 1 --limit 16 --run_name smoke_dual
python train_eval.py --data_root "%DATA%" --mode early --epochs 1 --limit 16 --run_name smoke_early
echo SMOKE TEST DONE - upar koi error to nahi aaya? check karo
pause
