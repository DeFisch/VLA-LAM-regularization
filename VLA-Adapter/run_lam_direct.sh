#!/bin/bash
# Train LAM prediction directly using autoregressive head with teacher forcing
# Output: 4 tokens × 16 classes (LAM codes)
# Uses LoRA to finetune VLM alongside LAM head

set -e

PYTHON="/home/daniel/code/miniconda3/envs/vla_lam/bin/python"

$PYTHON vla-scripts/train_lam_direct.py \
    --vlm_path "pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b" \
    --config_file_path "pretrained_models/configs" \
    --use_minivlm true \
    --lam_path "/home/daniel/code/lam-latent/checkpoints/univla-latent-action-model/lam-stage-2.ckpt" \
    --head_type "v3" \
    --hidden_dim 1024 \
    --num_layers 2 \
    --num_heads 8 \
    --use_lora true \
    --lora_rank 32 \
    --data_root_dir "/home/daniel/code/lam-latent/datasets/rlds/modified_libero_rlds" \
    --dataset_name "libero_10_no_noops" \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --max_steps 10000 \
    --wandb_project "vla-lam-direct" \
    --run_id_note "v3-ar-lora"
