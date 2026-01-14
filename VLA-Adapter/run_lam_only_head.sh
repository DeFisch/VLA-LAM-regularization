#!/bin/bash
# Train with L1RegressionActionHead + LAM auxiliary loss (LAM-only mode)
# Uses TransformerLAMDecoder for LAM prediction

set -e

PYTHON="/home/daniel/code/miniconda3/envs/vla_lam/bin/python"

WANDB_MODE=offline $PYTHON vla-scripts/finetune_lam.py \
    --vlm_path "pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b" \
    --config_file_path "pretrained_models/configs" \
    --use_minivlm true \
    --lam_path "/home/daniel/code/lam-latent/checkpoints/univla-latent-action-model/lam-stage-2.ckpt" \
    --use_lam true \
    --lam_only true \
    --lam_loss_weight 1.0 \
    --lam_tap_block 24 \
    --lam_decoder_type "transformer" \
    --use_lora true \
    --lora_rank 32 \
    --data_root_dir "/home/daniel/code/lam-latent/modified_libero_rlds" \
    --dataset_name "libero_10_no_noops" \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --max_steps 3000 \
    --image_aug true \
    --wandb_project "vla-lam-only" \
    --run_id_note "lam-transformer-decoder"
