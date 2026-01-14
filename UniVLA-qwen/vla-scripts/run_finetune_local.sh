#!/bin/bash
# Run finetune_libero.py with local paths

cd /home/daniel/code/lam-latent/UniVLA-qwen

export PYTHONPATH=/home/daniel/code/lam-latent/UniVLA-qwen:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

python vla-scripts/finetune_libero.py \
    --vla_path /home/daniel/code/lam-latent/VLA-Adapter-clean/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
    --lam_path /home/daniel/code/lam-latent/checkpoints/univla-latent-action-model/lam-stage-2.ckpt \
    --data_root_dir /home/daniel/code/lam-latent/modified_libero_rlds \
    --dataset_name libero_10_no_noops \
    --run_root_dir ./runs_local \
    --adapter_tmp_dir ./runs_local \
    --batch_size 2 \
    --max_steps 100 \
    --save_steps 50 \
    --learning_rate 1e-4 \
    --freeze_vla True \
    --use_lora False \
    --use_wandb False \
    --resume_vla_ckpt null \
    --resume_action_decoder_ckpt null
