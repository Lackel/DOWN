#!/usr/bin bash
for seed in 0 1 2
do 
    python down.py \
        --dataset hwu64 \
        --save_model_path model_mtp\
        --seed $seed \
        --save_model \
        --gpu_id 3
done