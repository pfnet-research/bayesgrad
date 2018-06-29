#!/usr/bin/env bash

set -eu

gpu=${1:--1}
# seed start to end
start=0
end=30
num_train=1000
unit_num=16
epoch=100
label=pyridine
method=ggnndrop
ratio=0.25

for ((seed=$start; seed < $end; seed++)); do
    python train_tox21.py -g ${gpu} --iterator-type=balanced --label=${label} --method=${method} --epoch=${epoch} --unit-num=${unit_num} --n-layers=1 -b 32 --conv-layers=4 --num-train=${num_train} --seed=${seed} --dropout-ratio=${ratio} --out=results/${method}_${label}_numtrain${num_train}_seed${seed}
done
