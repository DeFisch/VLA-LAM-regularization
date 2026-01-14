#!/bin/bash
# LAM-only training with TransformerLAMDecoder
# Uses VLM hidden states directly (vision patch features) for LAM prediction

set -e

PYTHON="/home/daniel/code/miniconda3/envs/vla_lam/bin/python"

$PYTHON vla-scripts/finetune_lam.py \
    --lam_path "/home/daniel/code/lam-latent/checkpoints/univla-latent-action-model/lam-stage-2.ckpt" \
    --lam_tap_block 18 \
    --lam_decoder_type "transformer" \
    --lam_use_vlm_hidden true \
    --lam_only true \
    --max_steps 5000 \
    --data_root_dir "/home/daniel/code/lam-latent/datasets/rlds/modified_libero_rlds" \
    --dataset_name "libero_10_no_noops" \
    --use_minivlm true \
    --vlm_path "pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b" \
    --config_file_path "pretrained_models/configs" \
    --use_proprio true \
    --run_id_note "vlm-hidden-lam"
