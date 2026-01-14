#!/bin/bash
# Train LAM prediction using VLM hidden states directly (not MLPResNet output)
# Uses earlier tap block to get richer visual features

set -e

PYTHON="/home/daniel/code/miniconda3/envs/vla_lam/bin/python"

$PYTHON vla-scripts/finetune_lam.py \
    --vlm_path "pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b" \
    --config_file_path "pretrained_models/configs" \
    --use_minivlm true \
    --lam_path "/home/daniel/code/lam-latent/checkpoints/univla-latent-action-model/lam-stage-2.ckpt" \
    --use_lam true \
    --use_lam_only_head false \
    --lam_only true \
    --lam_loss_weight 1.0 \
    --lam_tap_block 12 \
    --lam_decoder_type "transformer" \
    --lam_use_vlm_hidden true \
    --use_lora true \
    --lora_rank 32 \
    --data_root_dir "/home/daniel/code/lam-latent/datasets/rlds/modified_libero_rlds" \
    --dataset_name "libero_10_no_noops" \
    --batch_size 8 \
    --learning_rate 1e-3 \
    --max_steps 10000 \
    --wandb_project "vla-lam-only" \
    --run_id_note "vlm-hidden-tap12-lr1e3"
