@echo off
REM ===== PHASE 2 : seed 1 aur seed 2 (mean +- std ke liye). Sir ke bolne ke baad hi chalana =====
set DATA=C:\Users\aksha\Downloads\Major Project\CAMO-ViT\VIAC
set BS=4

for %%S in (1 2) do (
  python train_eval.py --data_root "%DATA%" --batch_size %BS% --mode dual --fusion gated  --seed %%S || echo dual_gated_s%%S >> failed.txt
  python train_eval.py --data_root "%DATA%" --batch_size %BS% --mode rgb   --seed %%S || echo rgb_s%%S >> failed.txt
  python train_eval.py --data_root "%DATA%" --batch_size %BS% --mode ir    --seed %%S || echo ir_s%%S >> failed.txt
  python train_eval.py --data_root "%DATA%" --batch_size %BS% --mode early --seed %%S || echo early_s%%S >> failed.txt
  python train_eval.py --data_root "%DATA%" --batch_size %BS% --mode dual --fusion add    --seed %%S || echo dual_add_s%%S >> failed.txt
  python train_eval.py --data_root "%DATA%" --batch_size %BS% --mode dual --fusion concat --seed %%S || echo dual_concat_s%%S >> failed.txt
)

python summarize.py
python make_figures.py --data_root "%DATA%"
python pack_results.py
echo PHASE 2 COMPLETE
pause
