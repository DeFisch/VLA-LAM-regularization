"""
finetune_lam_tokens.py

Train VLM to predict LAM tokens directly via next-token prediction (UniVLA style).

Instead of using a separate action head, we:
1. Add LAM tokens (<ACT_0> to <ACT_15>) to the tokenizer vocabulary
2. Convert LAM codes to these tokens in the input sequence
3. Train via standard cross-entropy loss on action token positions
4. Measure LAM token accuracy and sequence accuracy
"""

import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

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
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSDataset
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from prismatic.models import load, load_vla

# LAM imports
sys.path.insert(0, "/home/daniel/code/lam-latent/UniVLA")
from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel


# LAM token constants
NUM_LAM_TOKENS = 4
NUM_LAM_CLASSES = 16
LAM_TOKEN_NAMES = [f"<ACT_{i}>" for i in range(NUM_LAM_CLASSES)]
IGNORE_INDEX = -100


@dataclass
class TrainConfig:
    # Model
    vlm_path: str = "pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b"
    config_file_path: str = "pretrained_models/configs"
    use_minivlm: bool = True

    # LAM model
    lam_path: str = "/home/daniel/code/lam-latent/checkpoints/univla-latent-action-model/lam-stage-2.ckpt"
    lam_window_size: int = 12

    # Dataset
    data_root_dir: Path = Path("/home/daniel/code/lam-latent/datasets/rlds/modified_libero_rlds")
    dataset_name: str = "libero_10_no_noops"
    run_root_dir: Path = Path("runs")
    shuffle_buffer_size: int = 100_000
    image_aug: bool = True

    # Training
    batch_size: int = 8
    grad_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    max_steps: int = 10000
    warmup_steps: int = 500

    # LoRA
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0

    # Logging
    wandb_project: str = "vla-lam-tokens"
    wandb_entity: Optional[str] = None
    run_id_note: str = ""
    save_freq: int = 2500
    wandb_log_freq: int = 10

    # Debug
    seed: int = 42


def load_lam_model(lam_path: str, device: torch.device):
    """Load the LAM model for extracting latent codes."""
    print(f"Loading LAM model from {lam_path}...")
    lam_model = ControllableDINOLatentActionModel(
        in_dim=3,
        model_dim=768,
        latent_dim=128,
        num_latents=16,
        patch_size=14,
        enc_blocks=12,
        dec_blocks=12,
        num_heads=12,
        dropout=0.0,
    )

    checkpoint = torch.load(lam_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    # Remove "lam." prefix from keys
    new_state_dict = {k.replace("lam.", ""): v for k, v in state_dict.items()}
    lam_model.load_state_dict(new_state_dict, strict=True)
    lam_model = lam_model.to(device).eval()
    print("LAM model loaded successfully!")
    return lam_model


def add_lam_tokens_to_tokenizer(tokenizer):
    """Add LAM tokens (<ACT_0> to <ACT_15>) to the tokenizer."""
    # Check if tokens already exist
    existing = [t for t in LAM_TOKEN_NAMES if t in tokenizer.get_vocab()]
    if len(existing) == NUM_LAM_CLASSES:
        print(f"LAM tokens already exist in tokenizer")
        return tokenizer

    # Add new tokens
    num_added = tokenizer.add_tokens(LAM_TOKEN_NAMES, special_tokens=True)
    print(f"Added {num_added} LAM tokens to tokenizer")
    return tokenizer


def get_lam_token_ids(tokenizer):
    """Get the token IDs for LAM tokens."""
    token_ids = []
    for name in LAM_TOKEN_NAMES:
        ids = tokenizer.encode(name, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"LAM token {name} should encode to single token, got {ids}")
        token_ids.append(ids[0])
    return token_ids


class LAMBatchTransform:
    """Transform that converts LAM codes to token sequences for LLM prediction."""

    def __init__(
        self,
        tokenizer,
        image_transform,
        prompt_builder_fn,
        lam_model,
        lam_token_ids,
        use_minivlm=True,
        lam_window_size=12,
    ):
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.lam_model = lam_model
        self.lam_token_ids = lam_token_ids  # List of token IDs for <ACT_0> to <ACT_15>
        self.use_minivlm = use_minivlm
        self.lam_window_size = lam_window_size

    def __call__(self, rlds_batch):
        """Process a batch from RLDS dataset."""
        # Get images for LAM encoding
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]
        img = rlds_batch["observation"]["image_primary"]
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        # Get LAM window images
        lam_window = rlds_batch.get("lam_window", None)
        if lam_window is None:
            # Fallback: use current and next image
            lam_indices = torch.zeros(NUM_LAM_TOKENS, dtype=torch.long)
        else:
            # Extract LAM codes from image window
            with torch.no_grad():
                # lam_window: (window_size, H, W, C)
                window_tensor = torch.from_numpy(lam_window).float() / 255.0
                window_tensor = window_tensor.permute(0, 3, 1, 2)  # (T, C, H, W)

                # Resize to LAM input size (224x224)
                window_tensor = F.interpolate(window_tensor, size=(224, 224), mode="bilinear")

                # Get device from LAM model
                device = next(self.lam_model.parameters()).device
                window_tensor = window_tensor.to(device)

                # Encode with LAM (takes consecutive frame pairs)
                # Use first two frames for now
                if window_tensor.shape[0] >= 2:
                    frame1 = window_tensor[0:1]  # (1, C, H, W)
                    frame2 = window_tensor[1:2]  # (1, C, H, W)
                    lam_indices = self.lam_model.encode_to_indices(frame1, frame2)  # (1, 4)
                    lam_indices = lam_indices.squeeze(0).cpu()  # (4,)
                else:
                    lam_indices = torch.zeros(NUM_LAM_TOKENS, dtype=torch.long)

        # Build prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": ""},  # Will be filled with LAM tokens
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Get input text (without action tokens)
        input_text = prompt_builder.get_prompt()

        # Tokenize input
        input_ids = self.tokenizer(input_text, add_special_tokens=True).input_ids

        # Create action token sequence from LAM indices
        action_token_ids = [self.lam_token_ids[idx.item()] for idx in lam_indices]

        # Append action tokens to input
        full_input_ids = input_ids + action_token_ids

        # Create labels: mask everything except action tokens
        labels = [IGNORE_INDEX] * len(input_ids) + action_token_ids

        # Transform image
        if self.use_minivlm:
            pixel_values = self.image_transform(img)
        else:
            pixel_values = self.image_transform(img)

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(full_input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "lam_indices": lam_indices,  # Ground truth for accuracy computation
        }


def compute_lam_accuracy(logits, labels, lam_token_ids):
    """Compute LAM token accuracy and sequence accuracy."""
    # Find positions where labels are LAM tokens
    lam_token_set = set(lam_token_ids)

    batch_size = logits.shape[0]
    all_token_correct = []
    all_seq_correct = []

    for b in range(batch_size):
        # Find LAM token positions in this sample
        lam_positions = []
        for i, label in enumerate(labels[b]):
            if label.item() in lam_token_set:
                lam_positions.append(i)

        if len(lam_positions) != NUM_LAM_TOKENS:
            continue  # Skip if not exactly 4 LAM tokens

        # Get predictions at LAM positions
        sample_correct = []
        for pos in lam_positions:
            pred = logits[b, pos - 1].argmax().item()  # Predict next token
            target = labels[b, pos].item()
            sample_correct.append(pred == target)

        all_token_correct.extend(sample_correct)
        all_seq_correct.append(all(sample_correct))

    token_acc = sum(all_token_correct) / len(all_token_correct) if all_token_correct else 0.0
    seq_acc = sum(all_seq_correct) / len(all_seq_correct) if all_seq_correct else 0.0

    return token_acc, seq_acc


def train(cfg: TrainConfig):
    """Main training function."""
    print(f"Training LAM token prediction on {cfg.dataset_name}")

    # Setup distributed state
    distributed_state = PartialState()
    device = distributed_state.device

    # Setup run directory
    run_id = f"lam-tokens-{cfg.dataset_name}"
    if cfg.run_id_note:
        run_id = f"{run_id}-{cfg.run_id_note}"
    run_dir = cfg.run_root_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

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

    # Load processor and add LAM tokens
    print("Loading processor and adding LAM tokens...")
    processor = AutoProcessor.from_pretrained(cfg.config_file_path, trust_remote_code=True)
    processor.tokenizer = add_lam_tokens_to_tokenizer(processor.tokenizer)
    lam_token_ids = get_lam_token_ids(processor.tokenizer)
    print(f"LAM token IDs: {lam_token_ids}")

    # Load VLM
    print("Loading VLM...")
    if cfg.use_minivlm:
        vlm = load(cfg.vlm_path, hf_token='', load_for_training=True)

        config = AutoConfig.from_pretrained("pretrained_models/configs/config.json")
        vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16).to(device)

        # Rename keys
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

    # Resize token embeddings for new LAM tokens
    vla.resize_token_embeddings(len(processor.tokenizer))
    print(f"Resized token embeddings to {len(processor.tokenizer)}")

    # Apply LoRA
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

    # Create batch transform
    batch_transform = LAMBatchTransform(
        tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        lam_model=lam_model,
        lam_token_ids=lam_token_ids,
        use_minivlm=cfg.use_minivlm,
        lam_window_size=cfg.lam_window_size,
    )

    # Load dataset
    print("Loading dataset...")
    vla_config = vla.base_model.model.config if cfg.use_lora else vla.config
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

    # Optimizer
    optimizer = AdamW([p for p in vla.parameters() if p.requires_grad], lr=cfg.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.max_steps)

    # Training loop
    print(f"Starting training for {cfg.max_steps} steps...")
    progress = tqdm.tqdm(total=cfg.max_steps, desc="Training")
    recent_metrics = {
        "ce_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "lam_token_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "lam_seq_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
    }

    step = 0
    for batch in dataloader:
        if step >= cfg.max_steps:
            break

        # Forward pass
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = vla(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
                labels=batch["labels"].to(device),
            )

        loss = output.loss

        # Compute LAM accuracy
        with torch.no_grad():
            token_acc, seq_acc = compute_lam_accuracy(
                output.logits.float(),
                batch["labels"].to(device),
                lam_token_ids,
            )

        # Backward
        loss = loss / cfg.grad_accumulation_steps
        loss.backward()

        # Track metrics
        recent_metrics["ce_loss"].append(loss.item() * cfg.grad_accumulation_steps)
        recent_metrics["lam_token_accuracy"].append(token_acc)
        recent_metrics["lam_seq_accuracy"].append(seq_acc)

        # Optimizer step
        if (step + 1) % cfg.grad_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Logging
        if step % cfg.wandb_log_freq == 0 and distributed_state.is_main_process:
            avg_metrics = {k: sum(v) / len(v) for k, v in recent_metrics.items() if len(v) > 0}
            wandb.log({
                "train/ce_loss": avg_metrics.get("ce_loss", 0),
                "train/lam_token_accuracy": avg_metrics.get("lam_token_accuracy", 0),
                "train/lam_seq_accuracy": avg_metrics.get("lam_seq_accuracy", 0),
                "train/learning_rate": scheduler.get_last_lr()[0],
            }, step=step)

        # Save checkpoint
        if step > 0 and step % cfg.save_freq == 0:
            if cfg.use_lora:
                vla.save_pretrained(run_dir / f"checkpoint-{step}")
            print(f"Saved checkpoint at step {step}")

        progress.update(1)
        step += 1

    # Final save
    if cfg.use_lora:
        vla.save_pretrained(run_dir / "checkpoint-final")
    print(f"Training complete! Final checkpoint saved to {run_dir}")

    if distributed_state.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    cfg = draccus.parse(TrainConfig)
    train(cfg)
