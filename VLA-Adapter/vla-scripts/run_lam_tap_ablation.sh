#!/bin/bash
# LAM Tap Block Ablation Study
# Tests tapping hidden states at different MLP blocks for LAM prediction
# Each run: 1000 steps to compare LAM CE loss and accuracy

set -e

# Python path
PYTHON="/home/daniel/code/miniconda3/envs/vla_lam/bin/python"

# Common settings
MAX_STEPS=1000
LAM_PATH="/home/daniel/code/lam-latent/checkpoints/univla-latent-action-model/lam-stage-2.ckpt"
DATA_ROOT="/home/daniel/code/lam-latent/datasets/rlds/modified_libero_rlds"
DATASET="libero_10_no_noops"
VLM_PATH="pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b"
CONFIG_PATH="pretrained_models/configs"

echo "=========================================="
echo "LAM Tap Block Ablation Study"
echo "Testing blocks: 6, 12, 18, 24"
echo "Steps per run: $MAX_STEPS"
echo "Using mini VLM: $VLM_PATH"
echo "Dataset: $DATASET"
echo "=========================================="

# Block 6 - Early tap (more raw visual info, less action-specialized)
echo ""
echo "[1/4] Running with lam_tap_block=6..."
$PYTHON vla-scripts/finetune_lam.py \
    --lam_path "$LAM_PATH" \
    --lam_tap_block 6 \
    --max_steps $MAX_STEPS \
    --data_root_dir "$DATA_ROOT" \
    --dataset_name "$DATASET" \
    --use_minivlm true \
    --vlm_path "$VLM_PATH" \
    --config_file_path "$CONFIG_PATH" \
    --use_proprio true \
    --run_id_note "ablation-tap6"

# Block 12 - Middle tap
echo ""
echo "[2/4] Running with lam_tap_block=12..."
$PYTHON vla-scripts/finetune_lam.py \
    --lam_path "$LAM_PATH" \
    --lam_tap_block 12 \
    --max_steps $MAX_STEPS \
    --data_root_dir "$DATA_ROOT" \
    --dataset_name "$DATASET" \
    --use_minivlm true \
    --vlm_path "$VLM_PATH" \
    --config_file_path "$CONFIG_PATH" \
    --use_proprio true \
    --run_id_note "ablation-tap12"

# Block 18 - Late-middle tap
echo ""
echo "[3/4] Running with lam_tap_block=18..."
$PYTHON vla-scripts/finetune_lam.py \
    --lam_path "$LAM_PATH" \
    --lam_tap_block 18 \
    --max_steps $MAX_STEPS \
    --data_root_dir "$DATA_ROOT" \
    --dataset_name "$DATASET" \
    --use_minivlm true \
    --vlm_path "$VLM_PATH" \
    --config_file_path "$CONFIG_PATH" \
    --use_proprio true \
    --run_id_note "ablation-tap18"

# Block 24 - Final tap (current baseline, most action-specialized)
echo ""
echo "[4/4] Running with lam_tap_block=24..."
$PYTHON vla-scripts/finetune_lam.py \
    --lam_path "$LAM_PATH" \
    --lam_tap_block 24 \
    --max_steps $MAX_STEPS \
    --data_root_dir "$DATA_ROOT" \
    --dataset_name "$DATASET" \
    --use_minivlm true \
    --vlm_path "$VLM_PATH" \
    --config_file_path "$CONFIG_PATH" \
    --use_proprio true \
    --run_id_note "ablation-tap24"

echo ""
echo "=========================================="
echo "Ablation complete! Check wandb for results."
echo "Compare LAM CE loss and accuracy across runs."
echo "=========================================="
