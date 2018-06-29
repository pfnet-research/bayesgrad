#!/usr/bin/env bash

set -eu

gpu=${1:--1}
label=pyridine
seeds=30
num_train=1000
unit_num=16

# mode=relative
mode=absolute
method=ggnndrop
ratio=0.25

for scale in 0.05 0.10 0.15 0.20; do
    python calc_prcauc_with_seeds.py -g ${gpu} --label=${label} --unit-num=${unit_num} --n-layers=1 --dropout-ratio=${ratio} --num-train=${num_train} --seed=${seeds} --method=${method} --scale=${scale} --mode=${mode}
done
