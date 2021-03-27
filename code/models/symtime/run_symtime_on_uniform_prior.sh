#!/usr/bin/env bash

export TRAIN_FILE=../../../data/uniform-prior-symbolic-format/train.txt
export TEST_FILE=../../../data/uniform-prior-symbolic-format/test.txt

python3 train_t5_end.py \
    --output_dir=experiment_result \
    --model_type=t5 \
    --tokenizer_name=t5-large \
    --model_name_or_path=symtime-pretrained-model/start \
    --duration_model_path=symtime-pretrained-model/duration \
    --do_train \
    --do_eval \
    --num_train_epochs=50 \
    --train_data_file=$TRAIN_FILE \
    --eval_data_file=$TEST_FILE \
    --line_by_line \
    --per_gpu_train_batch_size=8 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps=4 \
    --per_device_eval_batch_size=8 \
    --per_gpu_eval_batch_size=8 \
    --save_steps=5000 \
    --logging_steps=10 \
    --overwrite_output_dir \
    --seed=10 \
