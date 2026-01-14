"""
finetune_libero_pathfix.py - UniVLA-qwen training with local path fixes only
"""
from datetime import timedelta
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.distributed as dist
import tqdm
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, BitsAndBytesConfig

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction_LIBERO
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransformLIBERO_withHis, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.models import load

import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from prismatic.models.policy.transformer_utils import MAPBlock
from transformers import PreTrainedModel, PretrainedConfig

class UniVLAHFConfig(PretrainedConfig):
    model_type = "qwen2"
    def __init__(self, vocab_size=151936, hidden_size=3072, tie_word_embeddings=False,
                 vision_backbone="dinosiglip-224", text_backbone="qwen2.5", **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.tie_word_embeddings = tie_word_embeddings
        self.vision_backbone = vision_backbone
        self.text_backbone = text_backbone

class UniVLAHFModel(PreTrainedModel):
    config_class = UniVLAHFConfig
    base_model_prefix = "qwen2"

    def __init__(self, config: UniVLAHFConfig, core: nn.Module):
        super().__init__(config)
        self.vla = core
        if hasattr(self.vla, "get_input_embeddings") and hasattr(self.vla, "get_output_embeddings"):
            if config.tie_word_embeddings:
                self.tie_weights()

    def forward(self, **kwargs):
        return self.vla(**kwargs)

    def get_input_embeddings(self):
        if hasattr(self.vla, "get_input_embeddings"):
            return self.vla.get_input_embeddings()
        for name, module in self.vla.named_modules():
            if isinstance(module, nn.Embedding) and ("embed_tokens" in name or "word_embeddings" in name):
                return module
        return None

    def set_input_embeddings(self, new_emb):
        if hasattr(self.vla, "set_input_embeddings"):
            return self.vla.set_input_embeddings(new_emb)
        raise NotImplementedError("set_input_embeddings not wired for this core model.")

    def get_output_embeddings(self):
        if hasattr(self.vla, "get_output_embeddings"):
            return self.vla.get_output_embeddings()
        head = getattr(self.vla, "lm_head", None) or getattr(self.vla, "classifier", None)
        return head

    def _init_weights(self, module):
        return

    @classmethod
    def from_vla(cls, vla_core: nn.Module, config: UniVLAHFConfig):
        return cls(config=config, core=vla_core)

# Single-process mode setup
using_dist = False
rank = 0
world_size = 1
local_rank = 0
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
device_id = local_rank
print(f"Using distributed: {using_dist}, rank: {rank}, world_size: {world_size}, local_rank: {local_rank}")

class ActionDecoder(torch.nn.Module):
    def __init__(self, window_size=12, hidden_dim=512, vis_dim=896):
        super().__init__()
        self.latent_action_pool = MAPBlock(n_latents=1, vis_dim=vis_dim, embed_dim=hidden_dim, n_heads=hidden_dim // 64)
        self.visual_pool = MAPBlock(n_latents=1, vis_dim=vis_dim, embed_dim=hidden_dim, n_heads=hidden_dim // 64)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, 7 * window_size), nn.Tanh())

    def forward(self, latent_action_tokens, visual_embed):
        visual_embed = self.visual_pool(visual_embed)
        latent_action_tokens = latent_action_tokens[:, -4:]
        action_token = self.latent_action_pool(latent_action_tokens, init_embed=visual_embed)
        action = self.proj(action_token)
        return action

class Wrapped_Model(torch.nn.Module):
    def __init__(self, vla, freeze_vla=True, window_size=12):
        super().__init__()
        self.vla = vla
        self.window_size = window_size
        self.action_decoder = ActionDecoder(window_size=window_size)
        if freeze_vla:
            self.vla.requires_grad_(False)

    def forward(self, batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vla_output = self.vla(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
                output_hidden_states=True,
            )
        loss, loss_one_step, latent_action_tokens = self.action_decoder_forward(batch, vla_output)
        return vla_output, loss, loss_one_step, latent_action_tokens

    def action_decoder_forward(self, batch, vla_output):
        P = self.vla.vision_backbone.dino_featurizer.patch_embed.num_patches
        visual_embed = vla_output.hidden_states[-1][:, 1:1+P, :].to(torch.float)
        latent_tokens = torch.cat([
            vla_output.hidden_states[-1][:, :1, :],
            vla_output.hidden_states[-1][:, 1+P:, :],
        ], dim=1)
        action_gt = batch["labels"].to(latent_tokens.device)
        mask = (action_gt >= 151665) & (action_gt <= 151680)

        latent_action_tokens = []
        for idx, per_sample_latent_tokens in enumerate(latent_tokens):
            per_sample_latent_action_tokens = per_sample_latent_tokens[mask[idx], :]
            latent_action_tokens.append(per_sample_latent_action_tokens)
        latent_action_tokens = torch.stack(latent_action_tokens).to(torch.float)

        pred_action = self.action_decoder(latent_action_tokens, visual_embed).reshape(-1, self.window_size, 7)
        loss = torch.nn.functional.l1_loss(pred_action, batch['actions'], reduction='none')
        loss_one_step = loss[:, 0].mean()
        loss = loss.mean()
        return loss, loss_one_step, latent_action_tokens

@dataclass
class FinetuneConfig:
    # Local paths
    vla_path: str = "/home/daniel/code/lam-latent/VLA-Adapter-clean/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b"
    lam_path: str = "/home/daniel/code/lam-latent/checkpoints/univla-latent-action-model/lam-stage-2.ckpt"
    data_root_dir: Path = Path("/home/daniel/code/lam-latent/modified_libero_rlds")
    dataset_name: str = "libero_10_no_noops"
    run_root_dir: Path = Path("runs_local")
    adapter_tmp_dir: Path = Path("runs_local")

    batch_size: int = 4
    max_steps: int = 5000
    save_steps: int = 1000
    learning_rate: float = 2e-5
    grad_accumulation_steps: int = 1
    image_aug: bool = True
    shuffle_buffer_size: int = 500
    save_latest_checkpoint_only: bool = True

    codebook_size: int = 16
    lam_model_dim: int = 768
    lam_latent_dim: int = 128
    lam_patch_size: int = 14
    lam_enc_blocks: int = 12
    lam_dec_blocks: int = 12
    lam_num_heads: int = 12
    window_size: int = 12

    freeze_vla: bool = False
    use_lora: bool = False
    lora_rank: int = 32
    lora_dropout: float = 0.0
    use_quantization: bool = False

    use_wandb: bool = False
    wandb_project: str = "finetune-LIBERO"
    wandb_entity: str = "danielfeng"
    run_id_note: Optional[str] = None

    resume_vla_ckpt: Optional[Path] = None
    resume_action_decoder_ckpt: Optional[Path] = None

@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    exp_id = (f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}+b{cfg.batch_size * cfg.grad_accumulation_steps}+lr-{cfg.learning_rate}")
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.image_aug:
        exp_id += "--image_aug"
    if cfg.freeze_vla:
        exp_id += f'=frozenVLA'
    exp_id += f'=w-LowLevelDecoder-ws-{cfg.window_size}'

    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run Directory: {run_dir}")

    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")

    # Load VLA model
    vla = load(cfg.vla_path, load_for_training=not cfg.freeze_vla)
    print(f"Loaded base VLM from {cfg.vla_path}")

    if cfg.resume_vla_ckpt is not None:
        print(f"Resuming VLA weights from checkpoint: {str(cfg.resume_vla_ckpt)}")
        vla_ckpt = torch.load(str(cfg.resume_vla_ckpt), map_location="cpu")
        vla.load_state_dict(vla_ckpt, strict=True)

    if cfg.freeze_vla:
        vla.eval()
        for param in vla.parameters():
            param.requires_grad = False
    else:
        vla.train()

    if cfg.use_lora:
        cfg_hf = UniVLAHFConfig(text_backbone="qwen2.5", vision_backbone="dino-siglip-224", tie_word_embeddings=False)
        vla = UniVLAHFModel.from_vla(vla, cfg_hf)
        vla.to(dtype=torch.bfloat16)

    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    if cfg.use_lora:
        all_linear = [name for name, mod in vla.named_modules() if isinstance(mod, nn.Linear)]
        lora_config = LoraConfig(r=cfg.lora_rank, lora_alpha=min(cfg.lora_rank, 16), lora_dropout=cfg.lora_dropout,
                                  target_modules=all_linear, init_lora_weights="gaussian")
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    tokenizer = vla.llm_backbone.get_tokenizer()
    img_transform = vla.vision_backbone.get_image_transform()
    action_tokenizer = ActionTokenizer(tokenizer)

    wrapped_model = Wrapped_Model(vla=vla, freeze_vla=cfg.freeze_vla, window_size=cfg.window_size).to(device_id)

    if cfg.resume_action_decoder_ckpt is not None:
        print(f"Resuming Action Decoder weights from checkpoint: {str(cfg.resume_action_decoder_ckpt)}")
        action_decoder_ckpt = torch.load(str(cfg.resume_action_decoder_ckpt), map_location="cpu")
        wrapped_model.action_decoder.load_state_dict(action_decoder_ckpt, strict=True)

    trainable_total_params = sum(p.numel() for p in wrapped_model.parameters() if p.requires_grad)
    print('Total Trainable Params: ', trainable_total_params)

    object.__setattr__(wrapped_model, 'module', wrapped_model)

    trainable_params = [param for param in wrapped_model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg.max_steps * 0.8), gamma=0.1)

    from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel

    latent_action_model = ControllableDINOLatentActionModel(
        in_dim=3, model_dim=cfg.lam_model_dim, latent_dim=cfg.lam_latent_dim, num_latents=cfg.codebook_size,
        patch_size=cfg.lam_patch_size, enc_blocks=cfg.lam_enc_blocks, dec_blocks=cfg.lam_dec_blocks,
        num_heads=cfg.lam_num_heads, dropout=0.,
    )

    lam_ckpt = torch.load(cfg.lam_path)['state_dict']
    new_ckpt = {key.replace("lam.", ""): lam_ckpt[key] for key in lam_ckpt.keys()}
    latent_action_model.load_state_dict(new_ckpt, strict=True)
    latent_action_model = latent_action_model.to(device_id).eval()

    batch_transform = RLDSBatchTransformLIBERO_withHis(
        latent_action_model, tokenizer, image_transform=img_transform,
        image_transform_lam=transforms.ToTensor(),
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
        window_size=cfg.window_size
    )

    vla_dataset = RLDSDataset(
        cfg.data_root_dir, cfg.dataset_name, batch_transform,
        resize_resolution=wrapped_model.module.vla.vision_backbone.default_image_resolution[1:],
        shuffle_buffer_size=cfg.shuffle_buffer_size, image_aug=cfg.image_aug,
        window_size=cfg.window_size + 1, training_phase='post-training',
    )

    if rank == 0:
        print("Saving Dataset Statistics...")
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    print(f"Rank {rank}: Creating DataLoader...")
    collator = PaddedCollatorForActionPrediction_LIBERO(tokenizer.model_max_length, tokenizer.pad_token_id, padding_side="right")
    dataloader = DataLoader(vla_dataset, batch_size=cfg.batch_size, sampler=None, collate_fn=collator, num_workers=0)

    if rank == 0 and cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)

    total_params = sum(p.numel() for p in wrapped_model.parameters())
    trainable_params = sum(p.numel() for p in wrapped_model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}, Trainable Parameters: {trainable_params}")

    print("\n" + "="*80)
    print("TRAINING STARTED - Monitoring LAM Token Accuracy")
    print("="*80 + "\n")

    with tqdm.tqdm(total=cfg.max_steps, leave=False, disable=True) as progress:
        wrapped_model.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            batch["input_ids"] = batch["input_ids"].to(device_id)
            batch["attention_mask"] = batch["attention_mask"].to(device_id)
            batch["labels"] = batch["labels"].to(device_id)
            for k in batch['pixel_values']:
                batch['pixel_values'][k] = batch['pixel_values'][k].to(device_id, dtype=torch.bfloat16) if isinstance(batch['pixel_values'][k], torch.Tensor) else batch['pixel_values'][k]
            batch['actions'] = batch['actions'].to(device_id)
            batch['latent_action_idx'] = batch['latent_action_idx'].to(device_id)

            output, act_loss, loss_one_step, latent_action_proj = wrapped_model(batch)
            loss = act_loss if cfg.freeze_vla else act_loss + output.loss

            normalized_loss = loss / cfg.grad_accumulation_steps
            torch.nn.utils.clip_grad_norm_(wrapped_model.parameters(), max_norm=1.)
            normalized_loss.backward()

            # Compute LAM Token Accuracy
            action_logits = output.logits[:, wrapped_model.module.vla.vision_backbone.dino_featurizer.patch_embed.num_patches:-1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = (action_gt >= 151665) & (action_gt <= 151680)
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())

            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)

            # Print metrics every 100 steps
            if rank == 0 and gradient_step_idx % 100 == 0:
                print(f"Step {gradient_step_idx:4d} | Loss: {smoothened_loss:.4f} | LAM Acc: {smoothened_action_accuracy*100:6.2f}% | VLA Loss: {output.loss.item():.4f} | Action L1: {act_loss.item():.4f}")

            if rank == 0 and cfg.use_wandb and gradient_step_idx % 5 == 0:
                wandb.log({
                    "train_loss": smoothened_loss,
                    "latent_action_accuracy": smoothened_action_accuracy,
                    "action_loss": act_loss.item(),
                    "action_loss_1step": loss_one_step.item(),
                    "lr": optimizer.state_dict()['param_groups'][0]['lr'],
                }, step=gradient_step_idx)

            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                progress.update()

            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if rank == 0:
                    print(f"\nSaving Model Checkpoint for Step {gradient_step_idx}")
                    save_dir = adapter_dir if cfg.use_lora else run_dir
                    if not cfg.freeze_vla:
                        if cfg.use_lora:
                            wrapped_model.module.vla.save_pretrained(str(save_dir) + f'/vla-adapter-{gradient_step_idx}')
                        else:
                            torch.save(wrapped_model.module.vla.state_dict(), str(save_dir) + f'/vla-full-{gradient_step_idx}.pt')
                    torch.save(wrapped_model.module.action_decoder.state_dict(), str(run_dir) + f'/action_decoder-{gradient_step_idx}.pt')

            if gradient_step_idx >= cfg.max_steps:
                print(f"\nMax step {cfg.max_steps} reached! Stopping training...")
                break

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    finetune()
