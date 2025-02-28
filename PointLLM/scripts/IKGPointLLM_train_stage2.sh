master_port=$((RANDOM % (65535 - 49152 + 1) + 49152))
# Get the filename without extension
filename=$(basename "$0" | cut -f 1 -d '.')

dir_path=PointLLM

model_name_or_path=/data/large/outputs/PointLLM_train_stage1/c17-epoch2 # Path to the output dir of stage 1 training
data_path=/data/large/ikg-data/instance_seg_c17_reflectivity_npy
anno_path=data/anno_data/ikgc17_complex_descriptions.json #ikgc17_complex_descriptions.json
output_dir=/data/large/outputs/PointLLM_train_stage2/c17-epoch2
#ds_path=/home/PointLLM/ds_config.json
#cd $dir_path
#    --deepspeed ds_config.json \
#    --gradient_checkpointing True 
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_config ./fsdp_config.json \
    #     --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \

PYTHONPATH=$dir_path:$PYTHONPATH \
torchrun --nnodes=1 --nproc_per_node=1 --master_port=$master_port pointllm/train/train_mem.py \
    --model_name_or_path $model_name_or_path \
    --data_path $data_path \
    --anno_path $anno_path \
    --output_dir $output_dir \
    --version v1 \
    --model_max_length 2048 \
    --num_train_epochs 3 \
    --per_device_train_batch_size  4\
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --eval_steps 100 \
    --save_strategy "no" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --bf16 True \
    --fix_llm False \
    --fix_pointnet True \
    --report_to wandb \
    --run_name $filename \
    --gradient_checkpointing True \
    --stage_2 True \
    --conversation_types "detailed_description" "single_round" "multi_round" \
    --use_color True \
    --deepspeed ./ds_config1.json \
    --gradient_checkpointing True