"""
train_lam_direct.py

Train VLA-Adapter to predict LAM codes directly (no continuous action prediction).

Architecture:
    VLM (Qwen 0.5B) -> Hidden States -> LAMPredictionHead -> 4 LAM tokens (each 0-15)

Loss: Cross-entropy on 4 token predictions
"""

import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import draccus
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from accelerate import PartialState
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
import wandb

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.lam_action_head import LAMPredictionHead, LAMPredictionHeadV2, LAMPredictionHeadV3, NUM_LAM_TOKENS, NUM_LAM_CLASSES
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, NUM_TOKENS
from prismatic.models import load, load_vla


# LAM imports
sys.path.insert(0, "/home/daniel/code/lam-latent/UniVLA")
from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel


@dataclass
class TrainConfig:
    # Model
    vlm_path: str = "pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b"
    config_file_path: str = "pretrained_models/configs"
    use_minivlm: bool = True

    # LAM model
    lam_path: str = "/home/daniel/code/lam-latent/checkpoints/univla-latent-action-model/lam-stage-2.ckpt"
    lam_window_size: int = 12

    # LAM head architecture
    head_type: str = "v3"  # "v1" (MLPResNet), "v2" (attention), "v3" (autoregressive with teacher forcing)
    hidden_dim: int = 1024
    num_layers: int = 2
    num_heads: int = 8

    # Dataset
    data_root_dir: Path = Path("/home/daniel/code/lam-latent/datasets/rlds/modified_libero_rlds")
    dataset_name: str = "libero_10_no_noops"
    run_root_dir: Path = Path("runs")
    shuffle_buffer_size: int = 100_000
    image_aug: bool = True

    # Training
    batch_size: int = 8
    grad_accumulation_steps: int = 1
    learning_rate: float = 5e-4
    max_steps: int = 10000
    warmup_steps: int = 500

    # LoRA (for finetuning VLM)
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0

    # Logging
    wandb_project: str = "vla-lam-direct"
    wandb_entity: Optional[str] = None
    run_id_note: str = ""
    save_freq: int = 2500
    wandb_log_freq: int = 10

    # Debug
    seed: int = 42


def load_lam_model(lam_path: str, device: torch.device):
    """Load pretrained LAM model for encoding frame pairs."""
    print(f"Loading LAM model from {lam_path}...")

    # Create model with standard parameters
    lam_model = ControllableDINOLatentActionModel(
        in_dim=3,  # image_channels
        model_dim=768,  # lam_model_dim
        latent_dim=128,  # lam_latent_dim
        num_latents=16,  # lam_num_latents (NUM_LAM_CLASSES)
        patch_size=14,  # lam_patch_size
        enc_blocks=12,  # lam_enc_blocks
        dec_blocks=12,  # lam_dec_blocks
        num_heads=12,  # lam_num_heads
        dropout=0.0,  # lam_dropout
    )

    # Load checkpoint
    lam_ckpt = torch.load(lam_path, map_location="cpu")["state_dict"]

    # Remove "lam." prefix from keys
    new_ckpt = {}
    for key in lam_ckpt.keys():
        new_ckpt[key.replace("lam.", "")] = lam_ckpt[key]

    lam_model.load_state_dict(new_ckpt, strict=True)
    lam_model = lam_model.to(device).eval()
    print("LAM model loaded successfully!")

    return lam_model




def get_unwrapped_model(model):
    """Get the underlying model from DDP wrapper."""
    if hasattr(model, "module"):
        return model.module
    return model


def run_forward_pass(vla, lam_head, batch, device):
    """Run forward pass and compute LAM prediction loss."""

    # Check if LAM targets are available
    if "lam_latent_indices" not in batch or batch["lam_latent_indices"] is None:
        raise ValueError("Batch does not contain LAM latent indices. Make sure dataset is configured with LAM model.")

    lam_targets = batch["lam_latent_indices"].to(device).long()  # (B, 4)

    # Get VLM hidden states
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output = vla(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
            labels=batch["labels"].to(device),
            output_hidden_states=True,
        )

    # Stack hidden states from all layers
    hidden_states = torch.stack(output.hidden_states, dim=1)  # (B, num_layers, seq_len, hidden_dim)

    # Predict LAM codes (pass targets for teacher forcing in V3)
    lam_logits = get_unwrapped_model(lam_head)(hidden_states, targets=lam_targets)  # (B, 4, 16)

    # Compute CE loss for each token position
    total_loss = 0.0
    for i in range(NUM_LAM_TOKENS):
        token_logits = lam_logits[:, i, :]  # (B, 16)
        token_targets = lam_targets[:, i]  # (B,)
        total_loss += F.cross_entropy(token_logits, token_targets)
    total_loss = total_loss / NUM_LAM_TOKENS

    # Compute accuracy
    lam_preds = lam_logits.argmax(dim=-1)  # (B, 4)
    token_acc = (lam_preds == lam_targets).float().mean()
    seq_acc = (lam_preds == lam_targets).all(dim=-1).float().mean()

    return total_loss, {
        "lam_ce_loss": total_loss.item(),
        "lam_token_accuracy": token_acc.item(),
        "lam_seq_accuracy": seq_acc.item(),
    }


def train(cfg: TrainConfig):
    """Main training loop."""
    print(f"Training LAM prediction on {cfg.dataset_name}")
    print(f"Head type: {cfg.head_type}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index if distributed_state.num_processes > 1 else 0

    # Create run directory
    run_id = f"lam-direct-{cfg.head_type}-{cfg.dataset_name}"
    if cfg.run_id_note:
        run_id += f"-{cfg.run_id_note}"
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    # Initialize wandb
    if distributed_state.is_main_process:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=run_id,
            config=vars(cfg),
        )

    # Load LAM model
    print("Loading LAM model...")
    lam_model = load_lam_model(cfg.lam_path, device)

    # Register custom model classes
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load VLM
    print("Loading VLM...")
    if cfg.use_minivlm:
        # Load mini VLM and transfer state dict
        vlm = load(cfg.vlm_path, hf_token='', load_for_training=False)

        config = AutoConfig.from_pretrained("pretrained_models/configs/config.json")
        vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16).to(device)

        # Rename keys to match VLA model
        replace_map = [
            ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
            ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
            ("llm_backbone.llm", "language_model"),
            ("projector.projector.0", "projector.fc1"),
            ("projector.projector.2", "projector.fc2"),
            ("projector.projector.4", "projector.fc3"),
            ("gamma", "scale_factor"),
        ]

        def rename_state_dict_keys(state_dict, replace_map):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                for old, new in replace_map:
                    if old in new_k:
                        new_k = new_k.replace(old, new)
                new_state_dict[new_k] = v
            return new_state_dict

        old_state_dict = vlm.state_dict()
        renamed_state_dict = rename_state_dict_keys(old_state_dict, replace_map)
        missing_keys, unexpected_keys = vla.load_state_dict(renamed_state_dict, strict=False)
        print(f"Loaded VLM: {len(missing_keys)} missing, {len(unexpected_keys)} unexpected keys")
        del vlm, old_state_dict
    else:
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.config_file_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=False,
        ).to(device)

    # Apply LoRA to VLM for finetuning
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=2 * cfg.lora_rank,
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()
        vla.train()
    else:
        # Freeze VLM if not using LoRA
        vla.eval()
        for param in vla.parameters():
            param.requires_grad = False

    # Create LAM prediction head
    base_vla = vla.base_model.model if cfg.use_lora else vla
    llm_dim = base_vla.llm_dim if hasattr(base_vla, 'llm_dim') else base_vla.config.hidden_size
    print(f"VLM hidden dim: {llm_dim}")

    if cfg.head_type == "v1":
        lam_head = LAMPredictionHead(
            input_dim=llm_dim,
            hidden_dim=cfg.hidden_dim,
            num_blocks=24,
            use_pro_version=True,
        ).to(torch.bfloat16).to(device)
    elif cfg.head_type == "v2":
        lam_head = LAMPredictionHeadV2(
            input_dim=llm_dim,
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
        ).to(torch.bfloat16).to(device)
    else:  # v3 - autoregressive with teacher forcing
        lam_head = LAMPredictionHeadV3(
            input_dim=llm_dim,
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
        ).to(torch.bfloat16).to(device)

    print(f"LAM head parameters: {sum(p.numel() for p in lam_head.parameters()):,}")

    # Load dataset with LAM model for extracting LAM codes
    print("Loading dataset...")
    processor = AutoProcessor.from_pretrained(cfg.config_file_path, trust_remote_code=True)

    # Create action tokenizer (needed for RLDSBatchTransform)
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Create batch transform with LAM model
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=False,
        use_proprio=False,
        use_minivlm=cfg.use_minivlm,
        use_lam=True,
        lam_model=lam_model,
        lam_window_size=cfg.lam_window_size,
    )

    # Get VLM image size
    vla_config = base_vla.config if hasattr(base_vla, 'config') else vla.config
    image_size = vla_config.image_sizes if hasattr(vla_config, 'image_sizes') else (224, 224)
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(image_size),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        use_lam=True,
        lam_window_size=cfg.lam_window_size,
    )

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=collator,
        num_workers=0,
    )

    # Optimizer (train VLM with LoRA and LAM head)
    trainable_params = list(lam_head.parameters())
    if cfg.use_lora:
        trainable_params += [p for p in vla.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.max_steps)

    # Training loop
    print(f"Starting training for {cfg.max_steps} steps...")
    progress = tqdm.tqdm(total=cfg.max_steps, desc="Training")
    recent_metrics = {
        "lam_ce_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "lam_token_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "lam_seq_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
    }

    step = 0
    for batch in dataloader:
        if step >= cfg.max_steps:
            break

        # Forward pass
        loss, metrics = run_forward_pass(vla, lam_head, batch, device)

        # Backward
        loss = loss / cfg.grad_accumulation_steps
        loss.backward()

        # Track metrics
        for k, v in metrics.items():
            recent_metrics[k].append(v)

        # Optimizer step
        if (step + 1) % cfg.grad_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Logging
        if step % cfg.wandb_log_freq == 0 and distributed_state.is_main_process:
            avg_metrics = {k: sum(v) / len(v) for k, v in recent_metrics.items() if len(v) > 0}
            wandb.log({
                "train/lam_ce_loss": avg_metrics.get("lam_ce_loss", 0),
                "train/lam_token_accuracy": avg_metrics.get("lam_token_accuracy", 0),
                "train/lam_seq_accuracy": avg_metrics.get("lam_seq_accuracy", 0),
                "train/learning_rate": scheduler.get_last_lr()[0],
            }, step=step)

        # Save checkpoint
        if step > 0 and step % cfg.save_freq == 0:
            ckpt_path = run_dir / f"checkpoint-{step}.pt"
            save_dict = {
                "step": step,
                "lam_head": lam_head.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(save_dict, ckpt_path)
            # Save VLM LoRA adapter separately
            if cfg.use_lora:
                vla.save_pretrained(run_dir / f"vla-adapter-{step}")
            print(f"Saved checkpoint to {ckpt_path}")

        progress.update(1)
        step += 1

    # Final save
    final_path = run_dir / "checkpoint-final.pt"
    torch.save({
        "step": step,
        "lam_head": lam_head.state_dict(),
    }, final_path)
    if cfg.use_lora:
        vla.save_pretrained(run_dir / "vla-adapter-final")
    print(f"Training complete! Final checkpoint: {final_path}")

    if distributed_state.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    cfg = draccus.parse(TrainConfig)
    train(cfg)
