#!/bin/sh

root=/data1/jxxiang/working
data_root=/data1/jxxiang/dataset
proj_dir=$root/DiffusionDialog_test/find_ori_2024

export CUDA_VISIBLE_DEVICES=2
export NLGEVAL_DATA=/data1/jxxiang/nlgeval_data
export TOKENIZERS_PARALLELISM=true

sent_stage_training=0
sent_stage_evaluation=1
sent_stage_test=0

# --load_checkpoint \
# --checkpoint /data1/jxxiang/working/DiffusionDialog/log/daily/2022-11-19.12-29-57.bartdiffusion/training_step_1500 \
    # --with_kl_loss \
    # --kl_loss_weight 1.0 \
    # --kl_target 5.0 \
        # > train.log 2>&1 &
    #> train_bartdiffusion.log 2>&1 &

    # --total_steps 16000 \
    # --logging_steps 20 \
    # --eval_steps 1000 \
    # --warmup_steps 1000 \
    # --grad_accum_steps 8 \
    # --train_batch_size 8 \
    # --dev_batch_size 8 \
    # --load_checkpoint \
    # --checkpoint /data1/jxxiang/working/DiffusionDialog/pretrain/checkpoints/2022-11-15.16-07-34.bartdiffusion/training_step_10000 \
if [ $sent_stage_training -eq 1 ]; then
    nohup python -u $proj_dir/src/train.py \
    --model bartdiffusion \
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
    --total_steps 10000 \
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
    > train_bartdiffusion_new.log 2>&1 &
fi

    # > eval.log 2>&1 &
if [ $sent_stage_evaluation -eq 1 ]; then
# evaluating
    nohup python -u $proj_dir/src/eval.py \
    --model bartdiffusion \
    --data daily \
    --eval_way nltk-eval \
    --tokenizer_path /data1/jxxiang/plm/bert-base-uncased \
    --train_set_split_name train \
    --dev_set_split_name dev \
    --test_set_split_name test \
    --save_result_dir /data1/jxxiang/working/DiffusionDialog_test/find_ori_2024/log/daily/2024-04-02.13-01-42.bartdiffusion \
    --dataset_path $data_root/DailyDialog \
    --log_dir $proj_dir/log \
    --checkpoint_path /data1/jxxiang/working/DiffusionDialog_test/find_ori_2024/log/daily/2024-04-02.13-01-42.bartdiffusion \
    --start_step 500 \
    --end_step 10000 \
    --interval_steps 500 \
    --max_source_length 256 \
    --max_target_length 128 \
    --max_context_turn 10 \
    --dev_batch_size 8 \
    --test_batch_size 8 \
    --beam_size 5 \
    --seed 2022 \
    > eval_bartdiffusion_new.log 2>&1 &
fi

if [ $sent_stage_test -eq 1 ]; then
    python $proj_dir/src/test.py \
    --model bartdiffusion \
    --data daily \
    --train_set_split_name train \
    --dev_set_split_name dev \
    --test_set_split_name test \
    --dataset_path $data_root/DailyDialog \
    --log_dir $proj_dir/log \
    --train_state diffusion \
    --log_dir $proj_dir/log \
    --max_source_length 256 \
    --max_target_length 128 \
    --max_context_turn 10 \
    --train_batch_size 8 \
    --dev_batch_size 8 \
    --beam_size 5 \
    --load_checkpoint \
    --checkpoint /data1/jxxiang/working/DiffusionDialog_2/log/daily/2022-12-03.22-56-22.bartdiffusion/training_step_10000 \
    --seed 2022
fi