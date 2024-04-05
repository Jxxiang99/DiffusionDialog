#!/bin/sh

root=/data1/jxxiang/working
data_root=/data1/jxxiang/dataset
proj_dir=$root/DiffusionDialog_test/find_ori_2024

export CUDA_VISIBLE_DEVICES=3
export NLGEVAL_DATA=/data1/jxxiang/nlgeval_data
export TOKENIZERS_PARALLELISM=true

sent_stage_training=0
sent_stage_evaluation=1
sent_stage_test=0

# --load_checkpoint \
# --checkpoint /home/jxxiang/working/LightDialog/log/daily/bartbase_daily_training_step_20 \
    # --with_kl_loss \
    # --kl_loss_weight 1.0 \
    # --kl_target 5.0 \
        # > train.log 2>&1 &
    #> train_bartdiffusion_persona.log 2>&1 &

    # --total_steps 16000 \
    # --logging_steps 20 \
    # --eval_steps 1000 \
    # --warmup_steps 1000 \
    # --grad_accum_steps 8 \
    # --train_batch_size 8 \
    # --dev_batch_size 8 \
if [ $sent_stage_training -eq 1 ]; then
    nohup python -u $proj_dir/src/train.py \
    --model bartdiffusion \
    --data persona \
    --train_set_split_name train \
    --dev_set_split_name dev \
    --test_set_split_name test \
    --dataset_path $data_root/PersonaChat \
    --plm_init_path /data1/jxxiang/plm/bart \
    --save_model_path $proj_dir/log \
    --log_dir $proj_dir/log \
    --max_source_length 256 \
    --max_target_length 128 \
    --max_context_turn 10 \
    --total_steps 20000 \
    --logging_steps 20 \
    --eval_steps 1000 \
    --warmup_steps 2000 \
    --grad_accum_steps 16 \
    --train_batch_size 8 \
    --dev_batch_size 8 \
    --optimizer adamw \
    --learning_rate 1e-4 \
    --clip_value 1 \
    --beam_size 5 \
    --seed 2022 \
    > train_bartdiffusion_persona_new.log 2>&1 &
fi

    # > eval.log 2>&1 &
if [ $sent_stage_evaluation -eq 1 ]; then
# evaluating
    nohup python -u $proj_dir/src/eval.py \
    --model bartdiffusion \
    --data persona \
    --eval_way nltk-eval \
    --tokenizer_path /data1/jxxiang/plm/bert-base-uncased \
    --train_set_split_name train \
    --dev_set_split_name dev \
    --test_set_split_name test \
    --save_result_dir /data1/jxxiang/working/DiffusionDialog_test/find_ori_2024/log/persona/2024-04-02.13-15-12.bartdiffusion \
    --dataset_path $data_root/PersonaChat \
    --log_dir $proj_dir/log \
    --checkpoint_path /data1/jxxiang/working/DiffusionDialog_test/find_ori_2024/log/persona/2024-04-02.13-15-12.bartdiffusion \
    --start_step 1000 \
    --end_step 20000 \
    --interval_steps 1000 \
    --max_source_length 256 \
    --max_target_length 128 \
    --max_context_turn 10 \
    --dev_batch_size 8 \
    --test_batch_size 8 \
    --beam_size 5 \
    --seed 2022 \
    > eval_bartdiffusion_persona_new.log 2>&1 &
fi

if [ $sent_stage_test -eq 1 ]; then
    python $proj_dir/src/test.py \
    --model bartdiffusion \
    --data persona \
    --train_set_split_name train \
    --dev_set_split_name dev \
    --test_set_split_name test \
    --load_checkpoint \
    --checkpoint /data1/jxxiang/working/DiffusionDialog/log/persona/2022-11-10.19-37-56.bartdiffusion/training_step_20000 \
    --dataset_path $data_root/PersonaChat \
    --train_state diffusion \
    --log_dir $proj_dir/log \
    --max_source_length 256 \
    --max_target_length 128 \
    --max_context_turn 10 \
    --dev_batch_size 8 \
    --beam_size 5 \
    --seed 2022
fi