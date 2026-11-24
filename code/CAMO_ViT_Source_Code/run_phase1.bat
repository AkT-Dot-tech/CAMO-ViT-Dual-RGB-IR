@echo off
REM ===== PHASE 1 : seed 0, sab configurations (approx 6-10 ghante) =====
REM Neeche DATA ka path apne VIAC folder ka daalo (jisme Traindata aur Testdata hain)
set DATA=C:\Users\aksha\Downloads\Major Project\CAMO-ViT\VIAC
set BS=4

python train_eval.py --data_root "%DATA%" --batch_size %BS% --mode dual --fusion gated --seed 0 || echo dual_gated_s0 >> failed.txt
python train_eval.py --data_root "%DATA%" --batch_size %BS% --mode rgb   --seed 0 || echo rgb_s0 >> failed.txt
python train_eval.py --data_root "%DATA%" --batch_size %BS% --mode ir    --seed 0 || echo ir_s0 >> failed.txt
python train_eval.py --data_root "%DATA%" --batch_size %BS% --mode early --seed 0 || echo early_s0 >> failed.txt
python train_eval.py --data_root "%DATA%" --batch_size %BS% --mode dual --fusion add    --seed 0 || echo dual_add_s0 >> failed.txt
python train_eval.py --data_root "%DATA%" --batch_size %BS% --mode dual --fusion concat --seed 0 || echo dual_concat_s0 >> failed.txt
python train_eval.py --data_root "%DATA%" --batch_size %BS% --mode rgb --freeze_encoder --seed 0 || echo rgb_frozen_s0 >> failed.txt

python summarize.py
python make_figures.py --data_root "%DATA%"
python pack_results.py
echo PHASE 1 COMPLETE
pause
