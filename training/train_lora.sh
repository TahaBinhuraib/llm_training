DATA_PATH="data/kaggle/train_data.json"
EVAL_DATA_PATH="data/kaggle/val_data.json"
PATH_TO_DEEPSPEED_CONFIG="./deepspeed_configs/deepspeed_config_s2.json"

deepspeed train.py \
    --model_name_or_path codellama/CodeLlama-7b-Python-hf  \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path $DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --output_dir ./checkpoints \
    --num_train_epochs 2 \
    --fp16 True \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "steps" \
    --eval_steps 50  \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --q_lora False \
    --deepspeed $PATH_TO_DEEPSPEED_CONFIG \
    --gradient_checkpointing True \
    --flash_attn False
