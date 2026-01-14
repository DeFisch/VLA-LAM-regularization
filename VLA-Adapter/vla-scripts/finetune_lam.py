"""
finetune_lam.py

Fine-tunes VLA with auxiliary LAM latent prediction loss.

This uses a unified architecture where both continuous actions and discrete LAM latents
are predicted from the SAME MLPResNet backbone:

Architecture:
    VLM (Qwen 0.5B) -> Multi-layer Hidden States (24 layers)
                          |
                    MLPResNet (24 blocks with cross-attention)
                          |
               +----------+----------+
               |                     |
            fc2 (existing)     AutoregressiveLAMDecoder (new)
               |                     |
        Continuous Actions    LAM 4-Token Code (each 0-15)
              (L1 loss)       P(z1|x)·P(z2|x,z1)·P(z3|x,z1,z2)·P(z4|x,z1,z2,z3)
                                    (CE loss with teacher forcing)

Total Loss = L1_loss + lam_loss_weight * LAM_CE_loss
"""

import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import draccus
import torch
import torch.distributed as dist
import torch.nn as nn

def safe_barrier():
    """Skip barrier if running single-process or if NCCL fails on unsupported GPU."""
    if dist.is_initialized() and dist.get_world_size() > 1:
        try:
            safe_barrier()
        except RuntimeError as e:
            if "no kernel image" in str(e):
                print(f"Warning: Skipping barrier due to unsupported GPU architecture")
            else:
                raise
import torch.nn.functional as F
import tqdm
from accelerate import PartialState
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
import wandb

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    NUM_TOKENS
)
from prismatic.vla.datasets import RLDSDataset, RLDSBatchTransform
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.models import load, load_vla


# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# LAM code constants
NUM_LAM_TOKENS = 4  # LAM produces 4 tokens
NUM_LAM_CLASSES = 16  # Each token is 0-15


@dataclass
class FinetuneConfig:
    # fmt: off
    config_file_path: str = "openvla/openvla-7b"
    vlm_path: str = "openvla/openvla-7b"
    use_minivlm: bool = False
    resum_vla_path: str = "openvla/openvla-7b"

    # LAM Model
    lam_path: str = ""                               # Path to LAM model checkpoint
    use_lam: bool = True                             # Enable LAM auxiliary loss
    lam_loss_weight: float = 0.5                     # Weight for LAM CE loss
    lam_window_size: int = 12                        # Window size for LAM extraction
    lam_tap_block: int = 24                          # Which MLP block to tap for LAM (1-24, lower = earlier)
    lam_only: bool = False                           # Train with LAM CE loss only (no L1 action loss)
    lam_decoder_type: str = "transformer"            # LAM decoder type: "simple" or "transformer"
    lam_use_vlm_hidden: bool = False                 # Use VLM hidden states instead of MLPResNet output for LAM

    # Dataset
    data_root_dir: Path = Path("datasets/rlds")
    dataset_name: str = "aloha_scoop_x_into_bowl"
    run_root_dir: Path = Path("runs")
    shuffle_buffer_size: int = 100_000

    # Algorithm and architecture
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps: int = 50
    use_film: bool = False
    num_images_in_input: int = 1
    use_proprio: bool = False
    phase1_path: str = "None"

    # Training configuration
    batch_size: int = 8
    learning_rate: float = 5e-4
    lr_warmup_steps: int = 0.1
    num_steps_before_decay: int = 100000
    grad_accumulation_steps: int = 1
    max_steps: int = 200000
    max_grad_norm: float = 1.0                       # Gradient clipping to prevent NaN
    use_val_set: bool = False
    val_freq: int = 10_000
    val_time_limit: int = 180
    save_freq: int = 10_000
    save_latest_checkpoint_only: bool = False
    resume: bool = False
    resume_step: Optional[int] = None
    image_aug: bool = True
    diffusion_sample_freq: int = 50

    # LoRA
    use_lora: bool = False
    lora_rank: int = 32
    lora_dropout: float = 0.0
    merge_lora_during_training: bool = False

    # Full Finetune
    use_fz: bool = False

    # Logging
    wandb_entity: str = "fengzhenyang47"
    wandb_project: str = "vla-lam-training"
    run_id_note: Optional[str] = None
    run_id_override: Optional[str] = None
    wandb_log_freq: int = 10

    # Version
    use_pro_version: bool = True
    phase: str = "Training"
    # fmt: on


def remove_ddp_in_checkpoint(state_dict) -> dict:
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_run_id(cfg) -> str:
    if cfg.run_id_override is not None:
        run_id = cfg.run_id_override
    elif cfg.resume:
        run_id = cfg.config_file_path.split("/")[-1]
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.config_file_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_lam:
            run_id += f"+lam-w{cfg.lam_loss_weight}-tap{cfg.lam_tap_block}-{cfg.lam_decoder_type}"
            if cfg.lam_use_vlm_hidden:
                run_id += "-vlm"
            if cfg.lam_only:
                run_id += "-LAMONLY"
        if cfg.use_fz:
            run_id += f"+frozen+dropout-{cfg.lora_dropout}"
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "--image_aug"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
    return run_id


def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)


def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> nn.Module:
    # Skip DDP for single GPU to avoid NCCL issues on unsupported architectures
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return module
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)


def get_unwrapped_model(model: nn.Module) -> nn.Module:
    """Get the underlying model from DDP wrapper or return as-is."""
    if hasattr(model, 'module'):
        return model.module
    return model


def count_parameters(module: nn.Module, name: str) -> None:
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> DDP:
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.resum_vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)
        print(f'Loaded {module_name} checkpoint!')

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)

    return wrap_ddp(module, device_id, find_unused_params)


def run_forward_pass(
    vla,
    action_head,
    proprio_projector,
    batch,
    action_tokenizer,
    device_id,
    use_l1_regression,
    use_proprio,
    use_film,
    num_patches,
    use_lam,
    lam_loss_weight,
    compute_diffusion_l1=False,
    use_pro_version=True,
    cfg=None,
    lam_only=False,
    step=0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Forward pass with auxiliary LAM latent prediction.

    Returns:
        tuple: (loss, metrics_dict)
    """
    metrics = {}

    # Get ground-truth action labels (truncate to NUM_ACTIONS_CHUNK to match model output)
    ground_truth_actions = batch["actions"][:, :NUM_ACTIONS_CHUNK].to(device_id).to(torch.bfloat16)

    # Get ground-truth LAM latent indices if available
    has_lam_targets = use_lam and "lam_latent_indices" in batch and batch["lam_latent_indices"] is not None
    if has_lam_targets:
        lam_latent_targets = batch["lam_latent_indices"].to(device_id)  # (B, num_lam_tokens)

    # VLA forward pass
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"].to(device_id),
            output_hidden_states=True,
            proprio=batch["proprio"].to(device_id) if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            noisy_actions=None,
            noisy_action_projector=None,
            diffusion_timestep_embeddings=None,
            use_film=use_film,
        )

    # Get action masks
    ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    # Build multi-layer hidden states for action head
    multi_layer_hidden_states = []
    batch_size = batch["input_ids"].shape[0]

    for item in output.hidden_states[0:]:
        text_hidden_states = item[:, num_patches:-1]
        actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(
            batch_size, 1, NUM_TOKENS, -1
        ).to(torch.bfloat16)
        task_latent_states = item[:, :num_patches].reshape(batch_size, 1, num_patches, -1)
        all_hidden_states = torch.cat((task_latent_states, actions_hidden_states), 2)
        multi_layer_hidden_states.append(all_hidden_states)

    multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim=1)

    # Predict actions (and optionally LAM logits)
    # LAM decoder taps hidden states at lam_tap_block (configurable for ablation)
    return_lam_logits = use_lam and has_lam_targets
    result = get_unwrapped_model(action_head).predict_action(
        multi_layer_hidden_states,
        proprio=batch["proprio"].to(device_id) if use_proprio else None,
        proprio_projector=proprio_projector if use_proprio else None,
        phase=cfg.phase,
        return_lam_logits=return_lam_logits,
        lam_targets=lam_latent_targets if has_lam_targets else None,  # Teacher forcing
    )

    if return_lam_logits:
        predicted_actions, lam_logits = result
    else:
        predicted_actions = result
        lam_logits = None

    # L1 loss for continuous actions
    l1_loss = F.l1_loss(predicted_actions, ground_truth_actions)

    # LAM CE loss
    # LAM produces 4 tokens (each 0-15), modeled autoregressively:
    # P(z1|x) · P(z2|x,z1) · P(z3|x,z1,z2) · P(z4|x,z1,z2,z3)
    # lam_logits: (B, 4, 16) - 4 conditional distributions from autoregressive decoder
    # lam_latent_targets: (B, 4) - target indices for each token position
    lam_ce_loss = torch.tensor(0.0, device=device_id)
    lam_accuracy = torch.tensor(0.0, device=device_id)
    lam_seq_accuracy = torch.tensor(0.0, device=device_id)

    if has_lam_targets and lam_logits is not None:
        # Compute CE loss for each of the 4 token positions and sum
        # lam_logits: (B, 4, 16), lam_latent_targets: (B, 4)
        total_lam_ce = 0.0
        for token_idx in range(NUM_LAM_TOKENS):
            # logits for this token position: (B, 16)
            token_logits = lam_logits[:, token_idx, :]
            # targets for this token position: (B,)
            token_targets = lam_latent_targets[:, token_idx]
            # CE loss for this position
            total_lam_ce = total_lam_ce + F.cross_entropy(token_logits, token_targets, reduction='mean')

        # Average over 4 token positions
        lam_ce_loss = total_lam_ce / NUM_LAM_TOKENS

        # Compute accuracy (all 4 tokens must match for full accuracy, or per-token)
        lam_preds = lam_logits.argmax(dim=-1)  # (B, 4)
        lam_accuracy = (lam_preds == lam_latent_targets).float().mean()  # Per-token accuracy
        # Sequence match accuracy: 1 if all 4 tokens match, 0 otherwise
        lam_seq_accuracy = (lam_preds == lam_latent_targets).all(dim=-1).float().mean()

        # Debug: Print predictions vs targets every 10 steps to check for mode collapse
        if step % 10 == 0:
            print(f"\n=== Step {step} LAM Debug ===")
            print(f"GT  : {lam_latent_targets.tolist()}")
            print(f"Pred: {lam_preds.tolist()}")
            # Count unique predictions to detect mode collapse
            unique_preds = set(tuple(p.tolist()) for p in lam_preds)
            print(f"Unique predictions in batch: {len(unique_preds)}/{lam_preds.shape[0]}")

    # Combined loss
    if lam_only:
        # Train with LAM CE loss only (no L1 action loss)
        total_loss = lam_ce_loss
    else:
        total_loss = l1_loss + lam_loss_weight * lam_ce_loss

    metrics.update({
        "loss_value": total_loss.item(),
        "l1_loss": l1_loss.item(),
        "lam_ce_loss": lam_ce_loss.item() if has_lam_targets else 0.0,
        "lam_accuracy": lam_accuracy.item() if has_lam_targets else 0.0,
        "lam_seq_accuracy": lam_seq_accuracy.item() if has_lam_targets else 0.0,
    })

    # Detailed L1 losses
    if use_l1_regression:
        ground_truth_curr_action = ground_truth_actions[:, 0]
        predicted_curr_action = predicted_actions[:, 0]
        ground_truth_next_actions = ground_truth_actions[:, 1:]
        predicted_next_actions = predicted_actions[:, 1:]
        curr_action_l1_loss = F.l1_loss(ground_truth_curr_action, predicted_curr_action)
        next_actions_l1_loss = F.l1_loss(ground_truth_next_actions, predicted_next_actions)

        if compute_diffusion_l1:
            print(f'curr L1: {curr_action_l1_loss.item():.4f}, LAM CE: {lam_ce_loss.item():.4f}, LAM acc: {lam_accuracy.item():.4f}, LAM seq: {lam_seq_accuracy.item():.4f}')

        metrics.update({
            "curr_action_l1_loss": curr_action_l1_loss.item(),
            "next_actions_l1_loss": next_actions_l1_loss.item(),
        })

    return total_loss, metrics


def compute_smoothened_metrics(metrics_deques) -> dict:
    smoothened_metrics = {}
    for name, deque in metrics_deques.items():
        if deque and len(deque) > 0:
            smoothened_metrics[name] = sum(deque) / len(deque)
    return smoothened_metrics


def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    log_dict = {}
    for name, value in metrics.items():
        if name == "loss_value":
            log_dict[f"{prefix}/Loss"] = value
        else:
            log_dict[f"{prefix}/{name.replace('_', ' ').title()}"] = value
    wandb_entity.log(log_dict, step=step)


def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    vla,
    processor,
    proprio_projector,
    action_head,
    train_dataset,
    distributed_state,
    new_state_dict,
) -> None:
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"

    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f"Saving Model Checkpoint for Step {log_step}")

    safe_barrier()

    if distributed_state.is_main_process:
        processor.save_pretrained(checkpoint_dir)

        if cfg.use_fz:
            get_unwrapped_model(vla).save_pretrained(checkpoint_dir)
        else:
            get_unwrapped_model(vla).save_pretrained(adapter_dir)

        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")

        if cfg.use_l1_regression and action_head is not None:
            torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")

        if cfg.use_film:
            torch.save(
                get_unwrapped_model(vla).vision_backbone.state_dict(), checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}"
            )

    safe_barrier()

    # Merge LoRA if applicable
    if cfg.use_lora and cfg.merge_lora_during_training:
        if cfg.use_minivlm:
            config = AutoConfig.from_pretrained("pretrained_models/configs/config.json")
            base_vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16)
            # Handle both DDP-wrapped and non-wrapped models
            state_dict = vla.state_dict()
            key = 'module.base_model.model.action_queries.weight' if 'module.base_model.model.action_queries.weight' in state_dict else 'base_model.model.action_queries.weight'
            new_state_dict['action_queries.weight'] = state_dict[key].cpu()
            missing_keys, unexpected_keys = base_vla.load_state_dict(new_state_dict, strict=False)
        else:
            base_vla = AutoModelForVision2Seq.from_pretrained(
                cfg.config_file_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False, trust_remote_code=False
            )

        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()

        if distributed_state.is_main_process:
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved merged model for Step {log_step} at: {checkpoint_dir}")

        safe_barrier()


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    """
    Fine-tune VLA with auxiliary LAM latent prediction loss.
    """
    global RAW_STATE_DICT

    assert not (cfg.use_l1_regression and cfg.use_diffusion), (
        "Cannot do both L1 regression and diffusion!"
    )

    cfg.config_file_path = cfg.config_file_path.rstrip("/")
    print(f"Fine-tuning VLA with LAM auxiliary loss on `{cfg.dataset_name}`")
    print(f"LAM loss weight: {cfg.lam_loss_weight}")
    print(f"LAM tap block: {cfg.lam_tap_block} (hidden states tapped at block {cfg.lam_tap_block} for LAM prediction)")
    print(f"LAM decoder type: {cfg.lam_decoder_type}")
    print(f"LAM use VLM hidden: {cfg.lam_use_vlm_hidden}")
    if cfg.lam_only:
        print("*** LAM-ONLY MODE: Training with LAM CE loss only (no L1 action loss) ***")

    run_id = get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    # GPU setup
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # Initialize wandb
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{run_id}", mode="online")

    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPROPRIO_DIM: {PROPRIO_DIM}\n"
        f"\tNUM_LAM_TOKENS: {NUM_LAM_TOKENS}\n"
        f"\tNUM_LAM_CLASSES: {NUM_LAM_CLASSES}"
    )

    # Load LAM model for latent extraction
    lam_model = None
    if cfg.use_lam and cfg.lam_path:
        print(f"Loading LAM model from {cfg.lam_path}...")
        # Use local symlink or add UniVLA path
        sys.path.insert(0, '/home/daniel/code/lam-latent/UniVLA')
        from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel

        # Parameters from checkpoint hyperparameters
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

        lam_ckpt = torch.load(cfg.lam_path)['state_dict']
        new_ckpt = {}
        for key in lam_ckpt.keys():
            new_ckpt[key.replace("lam.", "")] = lam_ckpt[key]

        lam_model.load_state_dict(new_ckpt, strict=True)
        lam_model = lam_model.to(device_id).eval()
        print("LAM model loaded successfully!")

    # Model setup
    if model_is_on_hf_hub(cfg.config_file_path):
        vla_download_path = snapshot_download(repo_id=cfg.config_file_path)
        cfg.config_file_path = vla_download_path
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    if distributed_state.is_main_process:
        update_auto_map(cfg.config_file_path)
        check_model_logic_mismatch(cfg.config_file_path)

    safe_barrier()

    # Load processor and VLA
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    processor = AutoProcessor.from_pretrained(cfg.config_file_path, trust_remote_code=True)

    if cfg.use_minivlm:
        hf_token = ''
        if 'prism-qwen25-extra-dinosiglip-224px-0_5b' in cfg.vlm_path:
            vlm = load(cfg.vlm_path, hf_token=hf_token, load_for_training=True)
        else:
            vlm = load_vla(cfg.vlm_path, hf_token=hf_token, load_for_training=True)

        config = AutoConfig.from_pretrained("pretrained_models/configs/config.json")
        vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16).to(device_id)

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
        RAW_STATE_DICT = rename_state_dict_keys(old_state_dict, replace_map)
        missing_keys, unexpected_keys = vla.load_state_dict(RAW_STATE_DICT, strict=False)
        del old_state_dict
    else:
        RAW_STATE_DICT = {}
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.config_file_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            trust_remote_code=False,
        ).to(device_id)

    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=2 * cfg.lora_rank,
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        for name, param in vla.named_parameters():
            if "action_queries" in name:
                param.requires_grad = True
        vla.print_trainable_parameters()
    else:
        for name, param in vla.named_parameters():
            if "action_queries" in name:
                param.requires_grad = True

    # FiLM setup
    if cfg.use_film:
        count_parameters(vla.vision_backbone, "vla.vision_backbone (original)")
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone,
            llm_dim=vla.llm_dim,
        )
        count_parameters(vla.vision_backbone, "vla.vision_backbone (post-wrap)")
        if cfg.resume:
            state_dict = load_checkpoint("vision_backbone", cfg.config_file_path, cfg.resume_step)
            vla.model.vision_backbone.load_state_dict(state_dict)
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    # Wrap VLA with DDP
    vla = wrap_ddp(vla, device_id, find_unused=True)

    # Proprio projector
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device_id,
            {"llm_dim": get_unwrapped_model(vla).llm_dim, "proprio_dim": PROPRIO_DIM},
            to_bf16=True,
        )

    # Action head with LAM latent prediction
    action_head = None
    if cfg.use_l1_regression:
        action_head = init_module(
            L1RegressionActionHead,
            "action_head",
            cfg,
            device_id,
            {
                "input_dim": get_unwrapped_model(vla).llm_dim,
                "hidden_dim": get_unwrapped_model(vla).llm_dim,
                "action_dim": ACTION_DIM,
                "use_pro_version": cfg.use_pro_version,
                "use_lam": cfg.use_lam,
                "lam_tap_block": cfg.lam_tap_block,  # Which MLP block to tap for LAM (1-24)
                "lam_decoder_type": cfg.lam_decoder_type,  # "simple" or "transformer"
                "lam_use_vlm_hidden": cfg.lam_use_vlm_hidden,  # Use VLM hidden states for LAM
            },
            to_bf16=True,
        )

    # Get number of vision patches
    NUM_PATCHES = get_unwrapped_model(vla).vision_backbone.get_num_patches() * get_unwrapped_model(vla).vision_backbone.get_num_images_in_input()

    # Optimizer setup
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if cfg.use_l1_regression and action_head is not None:
        trainable_params += [param for param in action_head.parameters() if param.requires_grad]
    if cfg.use_proprio and proprio_projector is not None:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]

    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    original_lr = optimizer.param_groups[0]["lr"]

    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],
        gamma=0.1,
    )

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Dataset setup
    use_wrist_image = cfg.num_images_in_input > 1
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
        use_minivlm=cfg.use_minivlm,
        use_lam=cfg.use_lam,
        lam_model=lam_model,
        lam_window_size=cfg.lam_window_size,
    )

    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(get_unwrapped_model(vla).config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        use_lam=cfg.use_lam,
        lam_window_size=cfg.lam_window_size,
    )

    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    # Dataloader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )
    print('Len of dataloader: ', len(dataloader))

    # Metrics tracking
    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "lam_ce_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "lam_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "lam_seq_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
    }

    # Training loop
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        if action_head is not None:
            action_head.train()

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            compute_diffusion_l1 = (
                cfg.use_l1_regression and batch_idx % cfg.diffusion_sample_freq == 0
            )

            loss, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=NUM_PATCHES,
                use_lam=cfg.use_lam,
                lam_loss_weight=cfg.lam_loss_weight,
                compute_diffusion_l1=compute_diffusion_l1,
                use_pro_version=cfg.use_pro_version,
                cfg=cfg,
                lam_only=cfg.lam_only,
                step=batch_idx,
            )

            normalized_loss = loss / cfg.grad_accumulation_steps
            normalized_loss.backward()

            for metric_name, value in metrics.items():
                if metric_name in recent_metrics:
                    recent_metrics[metric_name].append(value)

            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            smoothened_metrics = compute_smoothened_metrics(recent_metrics)

            log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx
            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)

            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            if distributed_state.is_main_process and gradient_step_idx % cfg.wandb_log_freq == 0:
                wandb.log(
                    {"VLA Train/Learning Rate": scheduler.get_last_lr()[0]},
                    step=log_step,
                )

            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                # Gradient clipping to prevent NaN from gradient explosion
                if cfg.max_grad_norm > 0:
                    all_params = []
                    for param_group in optimizer.param_groups:
                        all_params.extend(param_group['params'])
                    torch.nn.utils.clip_grad_norm_(all_params, cfg.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    action_head=action_head,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                    new_state_dict=RAW_STATE_DICT,
                )

            if log_step == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
