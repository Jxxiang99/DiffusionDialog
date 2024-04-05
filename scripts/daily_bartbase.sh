#!/bin/sh

root=/data1/jxxiang/working
data_root=/data1/jxxiang/dataset
proj_dir=$root/DiffusionDialog_test/find_ori_2024

export CUDA_VISIBLE_DEVICES=1
export NLGEVAL_DATA=/data1/jxxiang/nlgeval_data
export TOKENIZERS_PARALLELISM=true

sent_stage_training=0
sent_stage_evaluation=1
sent_stage_test=0


# if [ $sent_stage_training -eq 1 ]; then
# accelerate launch --num_cpu_threads_per_process 2 \
#     $proj_dir/src/train.py \
#     --model bart \
#     --data daily \
#     --train_set_split_name train \
#     --dev_set_split_name dev \
#     --test_set_split_name test \
#     --dataset_path $data_root/Dialog-data/DailyDialog \
#     --plm_init_path $data_root/plm/bart-base \
#     --save_model_path $data_root/LightDialog \
#     --log_dir $proj_dir/log \
#     --max_source_length 256 \
#     --max_target_length 128 \
#     --max_context_turn 10 \
#     --total_steps 6000 \
#     --logging_steps 20 \
#     --eval_steps 2000 \
#     --warmup_steps 2000 \
#     --grad_accum_steps 2 \
#     --train_batch_size 32 \
#     --dev_batch_size 32 \
#     --optimizer adamw \
#     --learning_rate 2e-5 \
#     --clip_value 1 \
#     --beam_size 5 \
#     --seed 2022
# fi

# if [ $sent_stage_evaluation -eq 1 ]; then
# # evaluating
# accelerate launch --num_cpu_threads_per_process 2 \
#     $proj_dir/src/eval.py \
#     --model bart \
#     --data daily \
#     --train_set_split_name train \
#     --dev_set_split_name dev \
#     --test_set_split_name test \
#     --save_result_dir $proj_dir/output \
#     --dataset_path $data_root/Dialog-data/DailyDialog \
#     --log_dir $proj_dir/log \
#     --checkpoint_path $data_root/LightDialog \
#     --start_step 2000 \
#     --end_step 6000 \
#     --interval_steps 2000 \
#     --max_source_length 256 \
#     --max_target_length 128 \
#     --max_context_turn 10 \
#     --dev_batch_size 16 \
#     --test_batch_size 16 \
#     --beam_size 5 \
#     --seed 2022
# fi

# --load_checkpoint \
# --checkpoint /home/jxxiang/working/LightDialog/log/daily/bartbase_daily_training_step_20 \
if [ $sent_stage_training -eq 1 ]; then
    nohup python -u $proj_dir/src/train.py \
    --model bartbase \
    --data daily \
    --train_set_split_name train \
    --dev_set_split_name dev \
    --test_set_split_name test \
    --dataset_path $data_root/DailyDialog \
    --plm_init_path /data1/jxxiang/plm/bart \
    --save_model_path $proj_dir/log \
    --log_dir $proj_dir/log \
    --max_source_length 256 \
    --max_target_length 128 \
    --max_context_turn 10 \
    --total_steps 20000 \
    --logging_steps 20 \
    --eval_steps 500 \
    --warmup_steps 500 \
    --grad_accum_steps 16 \
    --train_batch_size 8 \
    --dev_batch_size 8 \
    --optimizer adamw \
    --learning_rate 1e-4 \
    --clip_value 1 \
    --beam_size 5 \
    --seed 2022 \
    > train_bartbase_new.log 2>&1 &
fi

if [ $sent_stage_evaluation -eq 1 ]; then
# evaluating
    nohup python -u $proj_dir/src/eval.py \
    --model bartbase \
    --data daily \
    --eval_way nltk-eval \
    --tokenizer_path /data1/jxxiang/plm/bert-base-uncased \
    --train_set_split_name train \
    --dev_set_split_name dev \
    --test_set_split_name test \
    --save_result_dir $proj_dir/output/daily \
    --dataset_path $data_root/DailyDialog \
    --log_dir $proj_dir/log \
    --checkpoint_path /data1/jxxiang/working/DiffusionDialog_test/find_ori_2024/log/daily/2024-04-02.14-12-46.bartbase \
    --start_step 500 \
    --end_step 20000 \
    --interval_steps 500 \
    --max_source_length 256 \
    --max_target_length 128 \
    --max_context_turn 10 \
    --dev_batch_size 8 \
    --test_batch_size 8 \
    --beam_size 5 \
    --seed 2022 \
    > eval_bartbase_new.log 2>&1 &
fi

if [ $sent_stage_test -eq 1 ]; then
    python $proj_dir/src/test.py \
    --model bartbase \
    --data daily \
    --train_set_split_name train \
    --dev_set_split_name dev \
    --test_set_split_name test \
    --dataset_path $data_root/DailyDialog_test \
    --plm_init_path /data/jxxiang/plm/bart \
    --save_model_path $proj_dir/log \
    --log_dir $proj_dir/log \
    --max_source_length 256 \
    --max_target_length 128 \
    --max_context_turn 10 \
    --total_steps 60 \
    --logging_steps 3 \
    --eval_steps 3 \
    --warmup_steps 2 \
    --grad_accum_steps 1 \
    --train_batch_size 2 \
    --dev_batch_size 2 \
    --optimizer adamw \
    --learning_rate 2e-4 \
    --clip_value 1 \
    --beam_size 5 \
    --seed 2022
fi