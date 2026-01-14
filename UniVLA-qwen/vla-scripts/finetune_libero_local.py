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
# from mast3r.mast3r import model
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction_LIBERO
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransformLIBERO_withHis, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.models import load

import gc

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from prismatic.models.policy.transformer_utils import MAPBlock

# pip install transformers>=4.41 peft>=0.11
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

# 1) A config that HF knows how to save/load
class UniVLAHFConfig(PretrainedConfig):
    model_type = "qwen2"
    def __init__(
        self,
        vocab_size=151936,
        hidden_size=3072,
        tie_word_embeddings=False,
        vision_backbone="dinosiglip-224",
        text_backbone="qwen2.5",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.tie_word_embeddings = tie_word_embeddings
        self.vision_backbone = vision_backbone
        self.text_backbone = text_backbone

# 2) The thin wrapper that exposes HF hooks PEFT expects
class UniVLAHFModel(PreTrainedModel):
    config_class = UniVLAHFConfig
    base_model_prefix = "qwen2"  # affects save_pretrained weight nesting

    def __init__(self, config: UniVLAHFConfig, core: nn.Module):
        super().__init__(config)
        # your original torch.nn.Module
        self.vla = core

        # Optionally let HF tie word embeddings if the core exposes them
        if hasattr(self.vla, "get_input_embeddings") and hasattr(self.vla, "get_output_embeddings"):
            if config.tie_word_embeddings:
                self.tie_weights()

    # ---- Required HF methods/quality-of-life hooks ----
    def forward(self, **kwargs):
        # simply delegate; keep names compatible with your training code
        # e.g. images=..., input_ids=..., attention_mask=..., labels=...
        return self.vla(**kwargs)

    # (Optional but nice) expose embeddings for HF utils/PEFT
    def get_input_embeddings(self):
        if hasattr(self.vla, "get_input_embeddings"):
            return self.vla.get_input_embeddings()
        # Fallback: find a likely embedding module by name
        for name, module in self.vla.named_modules():
            if isinstance(module, nn.Embedding) and ("embed_tokens" in name or "word_embeddings" in name):
                return module
        return None

    def set_input_embeddings(self, new_emb):
        if hasattr(self.vla, "set_input_embeddings"):
            return self.vla.set_input_embeddings(new_emb)
        # If your core stores it at a known path, set it there instead.
        raise NotImplementedError("set_input_embeddings not wired for this core model.")

    def get_output_embeddings(self):
        if hasattr(self.vla, "get_output_embeddings"):
            return self.vla.get_output_embeddings()
        # Try common names (lm_head/classifier)
        head = getattr(self.vla, "lm_head", None) or getattr(self.vla, "classifier", None)
        return head

    # For save_pretrained/from_pretrained to round-trip weights
    def _init_weights(self, module):
        # No-op: weights already initialized in `core` you passed in
        return

    @classmethod
    def from_vla(cls, vla_core: nn.Module, config: UniVLAHFConfig):
        return cls(config=config, core=vla_core)

def reinit_dist(backend="nccl", new_master_port: int | None = None, timeout_sec=1800):
    # 1) Make sure all collectives are done
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        # 3) Destroy the current process group
        dist.destroy_process_group()

def _bootstrap_dist_env():
    """
    Force our own clean process group via env:// even if a library has
    already initialized torch.distributed. We:
      1) destroy the old PG (no barrier, because it's buggy),
      2) bump MASTER_PORT to a new value,
      3) init a fresh PG on the new port.
    """
    have_env = ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)

    print(
        f"[bootstrap] RANK={os.environ.get('RANK')} "
        f"WORLD_SIZE={os.environ.get('WORLD_SIZE')} "
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
        f"MASTER_ADDR={os.environ.get('MASTER_ADDR')} "
        f"MASTER_PORT={os.environ.get('MASTER_PORT')}",
        flush=True,
    )

    # 1) If some library already initialized, tear it down
    if dist.is_available() and dist.is_initialized():
        print("[bootstrap] Existing process group detected, destroying it...", flush=True)
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"[bootstrap] Failed to destroy existing PG: {e}", flush=True)

        # 2) Bump MASTER_PORT so our new PG uses a fresh TCPStore
        try:
            old_port = int(os.environ.get("MASTER_PORT", "29500"))
        except ValueError:
            old_port = 29500
        new_port = old_port + 10   # deterministic across ranks
        os.environ["MASTER_PORT"] = str(new_port)
        print(f"[bootstrap] Bumped MASTER_PORT from {old_port} to {new_port}", flush=True)

    # 3) Init our own PG if env vars exist
    if have_env:
        try:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=timedelta(minutes=30),
            )
            print("[bootstrap] torch.distributed initialized via env://", flush=True)
        except Exception as e:
            print(f"[bootstrap] init_process_group failed: {e}", flush=True)
    else:
        print("[bootstrap] No RANK/WORLD_SIZE; single-process mode.", flush=True)

    is_dist = dist.is_available() and dist.is_initialized()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return is_dist

using_dist = _bootstrap_dist_env()
rank = dist.get_rank() if using_dist else "0"
world_size = dist.get_world_size() if using_dist else 1
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
torch.cuda.set_device(local_rank)
device_id = local_rank
print(f"Using distributed: {using_dist}, rank: {rank}, world_size: {world_size}, local_rank: {local_rank}")

def flatten_prismatic_ckpt(ckpt, to_cpu=True, cast_dtype=None, fix_layerscale=False):
    """
    ckpt: either the whole torch.load(...) dict or the ckpt['model'] subdict
    to_cpu: move tensors to CPU (safer for loading)
    cast_dtype: e.g., torch.bfloat16 to shrink memory
    fix_layerscale: if your model expects '.scale_factor' instead of '.gamma'
    """
    if "model" in ckpt:
        ckpt = ckpt["model"]

    flat = {}
    for group in ("vision_backbone", "llm_backbone", "projector"):
        sub = ckpt.get(group, {})
        for k, v in sub.items():
            new_k = f"{group}.{k}"

            # Optional LayerScale rename
            if fix_layerscale:
                new_k = new_k.replace(".ls1.gamma", ".ls1.scale_factor") \
                             .replace(".ls2.gamma", ".ls2.scale_factor") \
                             .replace(".gamma", ".scale_factor")

            t = v
            if to_cpu and isinstance(t, torch.Tensor):
                t = t.cpu()
            if cast_dtype is not None and isinstance(t, torch.Tensor) and t.is_floating_point():
                t = t.to(cast_dtype)

            flat[new_k] = t
    return flat

class ActionDecoder(torch.nn.Module):
    def __init__(self, window_size = 12, hidden_dim = 512, vis_dim = 896):
        super().__init__()
        self.latent_action_pool = MAPBlock(n_latents = 1, vis_dim = vis_dim, embed_dim = hidden_dim, n_heads = hidden_dim // 64)
        self.visual_pool = MAPBlock(n_latents = 1, vis_dim = vis_dim, embed_dim = hidden_dim, n_heads = hidden_dim // 64)

        self.proj = nn.Sequential(
                                nn.Linear(hidden_dim, 7 * window_size),
                                nn.Tanh(),
                    )

    def forward(self, latent_action_tokens, visual_embed):
        visual_embed = self.visual_pool(visual_embed)
        latent_action_tokens = latent_action_tokens[:, -4:]
        action_token = self.latent_action_pool(latent_action_tokens, init_embed = visual_embed)

        action = self.proj(action_token)

        return action

class Wrapped_Model(torch.nn.Module):
    def __init__(self, vla, freeze_vla = True, window_size = 12):
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
                output_hidden_states = True,        # Return intermediate tokens of all layers
            )
        loss, loss_one_step, latent_action_tokens = self.action_decoder_forward(batch, vla_output)

        return vla_output, loss, loss_one_step, latent_action_tokens

    def action_decoder_forward(self, batch, vla_output):
        P = self.vla.vision_backbone.dino_featurizer.patch_embed.num_patches  # 256

        visual_embed  = vla_output.hidden_states[-1][:, 1:1+P, :].to(torch.float)   # (B, 256, D)
        latent_tokens = torch.cat([
            vla_output.hidden_states[-1][:, :1, :],          # keep BOS
            vla_output.hidden_states[-1][:, 1+P:, :],        # rest of text/action tokens
        ], dim=1)                                            # (B, T - 256, D)
        action_gt = batch["labels"].to(latent_tokens.device)
        # mask is when value between 151665 and 151680
        mask = (action_gt >= 151665) & (action_gt <= 151680)

        latent_action_tokens = []
        for idx, per_sample_latent_tokens in enumerate(latent_tokens):
            per_sample_latent_action_tokens = per_sample_latent_tokens[mask[idx], :] # shape: (num_valid_tokens, D)
            latent_action_tokens.append(per_sample_latent_action_tokens)
        latent_action_tokens = torch.stack(latent_action_tokens).to(torch.float)

        pred_action = self.action_decoder(latent_action_tokens, visual_embed).reshape(-1, self.window_size, 7)
        loss = torch.nn.functional.l1_loss(pred_action, batch['actions'], reduction='none')
        loss_one_step = loss[:,0].mean()
        loss = loss.mean()

        return loss, loss_one_step, latent_action_tokens



@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "/home/daniel/code/lam-latent/checkpoints/prism-qwen25-extra-dinosiglip-224px-0_5b"
    lam_path: str = "/home/daniel/code/lam-latent/checkpoints/univla-latent-action-model/lam-stage-2.ckpt"
    # Directory Paths
    data_root_dir: Path = Path("/home/daniel/code/lam-latent/modified_libero_rlds")
    dataset_name: str = "libero_10_no_noops"                   # Name of fine-tuning dataset
    run_root_dir: Path = Path("runs_local")                    # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("runs_local")                 # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 4                                        # Fine-tuning batch size (reduced for local testing)
    max_steps: int = 100                                       # Max number of fine-tuning steps
    save_steps: int = 50                                       # Interval for checkpoint saving
    learning_rate: float = 1e-4                                # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                           # Gradient accumulation steps
    image_aug: bool = True                                     # Whether to train with image augmentations
    shuffle_buffer_size: int = 500                             # Dataloader shuffle buffer size (reduced)
    save_latest_checkpoint_only: bool = True                   # Whether to save only one checkpoint per run
    # LAM setting
    codebook_size: int = 16
    lam_model_dim: int = 768
    lam_latent_dim: int = 128
    lam_patch_size: int = 14
    lam_enc_blocks: int = 12
    lam_dec_blocks: int = 12
    lam_num_heads: int = 12
    window_size: int = 12

    # LoRA Arguments
    freeze_vla: bool = False                                   # Train VLA to learn LAM tokens
    use_lora: bool = False                                     # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                        # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                  # Dropout applied to LoRA weights
    use_quantization: bool = False                             # Whether to 4-bit quantize VLA for LoRA fine-tuning

    # Tracking Parameters
    use_wandb: bool = False                                    # Disabled for local testing
    wandb_project: str = "finetune-LIBERO"                     # Name of W&B project to log to
    wandb_entity: str = "danielfeng"                           # Name of entity to log under
    run_id_note: Optional[str] = None                          # Extra note for logging, Weights & Biases

    resume_vla_ckpt: Optional[Path] = None                     # No resume for fresh start
    resume_action_decoder_ckpt: Optional[Path] = None          # No resume for fresh start



@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    # distributed_state = PartialState()
    torch.cuda.set_device(device_id := int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"
    if cfg.freeze_vla:
        exp_id += f'=frozenVLA'

    exp_id += f'=w-LowLevelDecoder-ws-{cfg.window_size}'
   
    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run Directory: {run_dir}")

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    # AutoConfig.register("openvla", OpenVLAConfig)
    # AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    # AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    # AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # # Load OpenVLA Processor and Model using HF AutoClasses
    # processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    # vla = AutoModelForVision2Seq.from_pretrained(
    #     cfg.vla_path,
    #     torch_dtype=torch.bfloat16,
    #     quantization_config=quantization_config,
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True,
    # )
    # Load base VLM model from configurable path
    vla = load(cfg.vla_path, load_for_training=not cfg.freeze_vla)

    # Skip loading pre-trained UniVLA weights for local testing - just use base prism model
    print(f"Loaded base VLM from {cfg.vla_path}")

    if cfg.resume_vla_ckpt is not None:
        print(f"Resuming VLA weights from checkpoint: {str(cfg.resume_vla_ckpt)}")
        vla_ckpt = torch.load(str(cfg.resume_vla_ckpt), map_location="cpu")
        vla.load_state_dict(vla_ckpt, strict=True)

    if cfg.freeze_vla:
        vla.eval()
        for param in vla.parameters():
            assert not param.requires_grad
    else:
        if not cfg.use_lora:
            vla.train()
            for param in vla.parameters():
                assert param.requires_grad
    if cfg.use_lora:
        cfg_hf = UniVLAHFConfig(
            # Fill anything your training loop might want to save in config.json
            text_backbone="qwen2.5",
            vision_backbone="dino-siglip-224",
            tie_word_embeddings=False
        )
        vla = UniVLAHFModel.from_vla(vla, cfg_hf)
        vla.to(dtype=torch.bfloat16)

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        all_linear = [name for name, mod in vla.named_modules() if isinstance(mod, nn.Linear)]
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules=all_linear,
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    tokenizer = vla.llm_backbone.get_tokenizer()
    img_transform = vla.vision_backbone.get_image_transform()
    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(tokenizer)

    wrapped_model = Wrapped_Model(vla = vla, freeze_vla = cfg.freeze_vla, window_size=cfg.window_size).to(device_id)

    if cfg.resume_action_decoder_ckpt is not None:
        print(f"Resuming Action Decoder weights from checkpoint: {str(cfg.resume_action_decoder_ckpt)}")
        action_decoder_ckpt = torch.load(str(cfg.resume_action_decoder_ckpt), map_location="cpu")
        wrapped_model.action_decoder.load_state_dict(action_decoder_ckpt, strict=True)

    
    trainable_total_params = sum(p.numel() for p in wrapped_model.parameters() if p.requires_grad)
    print('Total Trainable Params: ', trainable_total_params)
    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    if using_dist:
        wrapped_model = DDP(wrapped_model, device_ids=[device_id], find_unused_parameters=not cfg.freeze_vla, gradient_as_bucket_view=True)
    else:
       object.__setattr__(wrapped_model, 'module', wrapped_model)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in wrapped_model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = int(cfg.max_steps * 0.8), gamma=0.1)

        
    from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel

    latent_action_model = ControllableDINOLatentActionModel(
        in_dim=3,
        model_dim=cfg.lam_model_dim,
        latent_dim=cfg.lam_latent_dim,
        num_latents=cfg.codebook_size,
        patch_size=cfg.lam_patch_size,
        enc_blocks=cfg.lam_enc_blocks,
        dec_blocks=cfg.lam_dec_blocks,
        num_heads=cfg.lam_num_heads,
        dropout=0.,
    )

    lam_ckpt = torch.load(cfg.lam_path)['state_dict']
    new_ckpt = {}
    for key in lam_ckpt.keys():
        new_ckpt[key.replace("lam.", "")] = lam_ckpt[key]

    latent_action_model.load_state_dict(new_ckpt, strict=True)
    latent_action_model = latent_action_model.to(device_id).eval()
    
    batch_transform = RLDSBatchTransformLIBERO_withHis(
        latent_action_model,
        tokenizer,
        image_transform=img_transform,
        image_transform_lam=transforms.ToTensor(),
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
        window_size=cfg.window_size
    )
    

    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=wrapped_model.module.vla.vision_backbone.default_image_resolution[1:],
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        window_size=cfg.window_size + 1,        # for constructing history latent actions
        training_phase='post-training',
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if rank == 0:
        print("Saving Dataset Statistics...")
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    print(f"Rank {rank}: Creating DataLoader...")
    collator = PaddedCollatorForActionPrediction_LIBERO(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Initialize Logging =>> W&B
    if rank == 0 and cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # double check how many parameters are trainable in wrapped_model
    total_params = sum(p.numel() for p in wrapped_model.parameters())
    trainable_params = sum(p.numel() for p in wrapped_model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}, Trainable Parameters: {trainable_params}")
    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
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

            # print("\n\n<----------input_ids---------->")
            # print(tokenizer.decode(batch["input_ids"][0],
            #         skip_special_tokens=False,
            #         clean_up_tokenization_spaces=True,))
            
            # labels_pre_decoding = batch["labels"][0]
            # # filter out all -100 labels
            # labels_pre_decoding = labels_pre_decoding[labels_pre_decoding != -100]
            # print("\n\n<----------labels---------->")
            # print(tokenizer.decode(labels_pre_decoding,
            #         skip_special_tokens=False,
            #         clean_up_tokenization_spaces=True,))
            
            # Forward pass
            output, act_loss, loss_one_step, latent_action_proj = wrapped_model(batch)
            loss = act_loss if cfg.freeze_vla else act_loss + output.loss

            # print("\n\n<----------output logits---------->")
            # print(tokenizer.decode(output.logits.argmax(dim=-1)[0],
            #         skip_special_tokens=False,
            #         clean_up_tokenization_spaces=True,))

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps
            torch.nn.utils.clip_grad_norm_(wrapped_model.parameters(), max_norm=1.)

            # Backward pass
            normalized_loss.backward()

            # Compute Accuracy and L1 Loss for Logging
            action_logits = output.logits[:, wrapped_model.module.vla.vision_backbone.dino_featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            # mask is when value between 151665 and 151680
            mask = (action_gt >= 151665) & (action_gt <= 151680)

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Compute sequence accuracy and show predictions every 50 steps
            batch_size = action_preds.shape[0]
            seq_correct = 0
            for i in range(batch_size):
                sample_mask = mask[i]
                if sample_mask.sum() >= 4:
                    pred_lam = action_preds[i][sample_mask][:4]
                    gt_lam = action_gt[i][sample_mask][:4]
                    if (pred_lam == gt_lam).all():
                        seq_correct += 1
            seq_accuracy = seq_correct / batch_size

            # Debug predictions every 50 steps
            step_idx = batch_idx // cfg.grad_accumulation_steps
            if step_idx % 50 == 0 and step_idx > 0:
                print(f"\n=== DEBUG Step {step_idx} - Predictions ===")
                for i in range(min(4, batch_size)):
                    sample_mask = mask[i]
                    if sample_mask.sum() >= 4:
                        pred_lam = action_preds[i][sample_mask][:4]
                        gt_lam = action_gt[i][sample_mask][:4]
                        pred_rel = (pred_lam - 151665).tolist()
                        gt_rel = (gt_lam - 151665).tolist()
                        print(f"  Sample {i}: pred={pred_rel}, gt={gt_rel}")
                print(f"  Seq Accuracy: {seq_accuracy*100:.2f}%")

            # Store recent train metrics
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            #   =>> Equal to current step metrics when not using gradient accumulation
            #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)

            # Print metrics every 10 steps for monitoring (rank==0 or "0" for non-distributed)
            if (rank == 0 or rank == "0") and gradient_step_idx % 10 == 0:
                import sys
                print(f"\n>>> Step {gradient_step_idx}: Loss={smoothened_loss:.4f}, LAM Acc={smoothened_action_accuracy*100:.2f}%, Seq Acc={seq_accuracy*100:.2f}%, Action L1={act_loss.item():.4f}", flush=True)
                sys.stdout.flush()

            # Push Metrics to W&B (every 5 gradient steps)
            if rank == 0 and cfg.use_wandb and gradient_step_idx % 5 == 0:

                wandb.log(
                    {
                        "train_loss": smoothened_loss,
                        "latent_action_accuracy": smoothened_action_accuracy,
                        "action_loss": act_loss.item(),
                        "action_loss_1step": loss_one_step.item(),
                        "lr": optimizer.state_dict()['param_groups'][0]['lr'],
                    },
                    step=gradient_step_idx,
                )

            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                progress.update()

            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if rank == 0:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                    # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                    save_dir = adapter_dir if cfg.use_lora else run_dir

                    # Save Processor & Weights
                    if not cfg.freeze_vla:
                        if cfg.use_lora:
                            # save only adapter weights
                            wrapped_model.module.vla.save_pretrained(str(save_dir) + f'/vla-adapter-{gradient_step_idx}')
                        else:
                            # save wrapped_model.module.vla as nn.Module
                            torch.save(wrapped_model.module.vla.state_dict(), str(save_dir) + f'/vla-full-{gradient_step_idx}.pt')

                    # Save low-level policy
                    torch.save(wrapped_model.module.action_decoder.state_dict(), str(run_dir) + f'/action_decoder-{gradient_step_idx}.pt')

                # Wait for processor and adapter weights to be saved by main process
                if using_dist:
                    dist.barrier()

            # Stop training when max_steps is reached
            if gradient_step_idx >= cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
