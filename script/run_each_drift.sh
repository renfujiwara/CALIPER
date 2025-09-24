#!/bin/bashz
online_learning='full'
i=1
n=1
bsz=1
bs=32
seq_len=30
pred_len=30
log=0
num_works=-1
workers=0
num_buffer=1
scale=1
epoch=50
train_rate=0.15
test_rate=0.8

today=$(TZ=UTC-9 date '+%Y%m%d')

#                   method_id
methods=(
    'Transformer'        # 0
    'MLP'                # 1
    'KernelRidge'        # 2
)


dataset=(
    'Dysts'           # 0
    'TEP'             # 1
    'MoCap'           # 2
)

detectors=(
    'ADWIN'         # 0
    'KSWIN'         # 1
)

Feature='M'

for method_id in 0 #1 2
do
method=${methods[$method_id]}

for data_id in 0 #1 2
do
if [ ${data_id} -eq 9 ]; then
    F='MS'
else
    F=${Feature}
fi
dataname=${dataset[$data_id]}

for detector_id in 0 #1
do
detector=${detectors[$detector_id]}

# use CALIPER
poetry run python -u main.py --method $method \
                             --task_name 'long_term_forecast' \
                             --detector $detector\
                             --fix_win 128 \
                             --scale $scale \
                             --drift_detection \
                             --use_CALIPER \
                             --CALIPER_njob 16 \
                             --root_path ./dataset \
                             --verbose $log \
                             --grid_search 1 \
                             --n_inner  $n \
                             --test_bsz $bsz \
                             --batch_size $bs\
                             --data $dataname \
                             --features $F \
                             --seq_len $seq_len \
                             --label_len 0 \
                             --pred_len $pred_len \
                             --train_rate $train_rate \
                             --test_rate $test_rate \
                             --des 'Exp' \
                             --train_epochs $epoch \
                             --num_buffer $num_buffer \
                             --num_workers $workers \
                             --num_works $num_works \
                             --online_learning $online_learning \
                             --suffix $today \
                             --use_adbfgs > $logfn 2>&1


done

done 
done








