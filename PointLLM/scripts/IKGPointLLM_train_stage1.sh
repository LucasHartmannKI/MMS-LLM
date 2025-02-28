master_port=$((RANDOM % (65535 - 49152 + 1) + 49152))
# Get the filename without extension
filename=$(basename "$0" | cut -f 1 -d '.')

dir_path=PointLLM
#model_name_or_path=/data/large/checkpoint/PointLLM_7B_v1.1_init
#data_path=/data/large/ikg-data/instance_seg_c17_reflectivity_npy
#anno_path=data/anno_data/ikgc17_brief_description_filtered.json # or PointLLM_brief_description_660K.json (including val sets)
#output_dir=outputs/PointLLM_train_stage1/$filename
#point_backbone_ckpt=/data/large/checkpoint/PointLLM_7B_v1.1_init/ckpt-epoch-002.pth #ckpt-best_vote.pth#point_bert_v1.2.pt

#cd $dir_path

export CUDA_VISIBLE_DEVICES=0 ##########
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64

#PYTHONPATH=$dir_path:$PYTHONPATH python  pointllm/train/train_mem.py \ 
torchrun --nnodes=1 --nproc_per_node=1 --master_port=$master_port pointllm/train/train_mem1.py \
    --model_name_or_path '/data/large/checkpoint/PointLLM_7B_v1.1_init' \
    --data_path '/data/large/ikg-data/instance_seg_c17_reflectivity_npy' \
    --anno_path 'data/anno_data/ikgc17_brief_description_filter.json' \
    --output_dir '/data/large/outputs/PointLLM_train_stage1/c17-epoch2' \
    --version v1 \
    --model_max_length 2048 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --evaluation_strategy "no" \
    --fix_llm True \
    --fix_pointnet True \
    --gradient_checkpointing True \
    --report_to wandb \
    --run_name $filename \
    --point_backbone_ckpt '/data/large/checkpoint/PointLLM_7B_v1.1_init/ckpt-epoch-002.pth' \
    --use_color True