"""Utils for evaluating robot policies in various environments."""

import os
from pyexpat import model
import random
import time

import numpy as np
import torch
import torch.nn as nn

from experiments.robot.openvla_utils import (
    get_vla,
    get_vla_action,
    get_vla_latent_action,
)

from prismatic.models import load
from prismatic.models.policy.transformer_utils import MAPBlock

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

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_model(cfg, wrap_diffusion_policy_for_droid=False):
    """Load model for evaluation."""
    if cfg.model_family == "openvla":
        model = get_vla(cfg)
    elif cfg.model_family == "qwen2":
        model = get_qwenvla(cfg)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    print(f"Loaded model: {type(model)}")
    return model

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


def get_qwenvla(cfg):
    vla = load('/fs/scratch/PAS2099/danielf/geometry_grounded_latents/UniVLA/checkpoints/prism-qwen25-extra-dinosiglip-224px-0_5b', load_for_training=False)
    trained_model = cfg.pretrained_checkpoint
    raw = torch.load(trained_model, map_location="cpu")
    # raw = flatten_prismatic_ckpt(raw, to_cpu=True, cast_dtype=torch.bfloat16, fix_layerscale=False)

    missing_keys, unexpected_keys = vla.load_state_dict(raw, strict=False)
    assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"
    assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"
    vla.eval()
    for param in vla.parameters():
        assert not param.requires_grad
    vla = vla.to(DEVICE)

    if cfg.adapter_checkpoint is not None:
        from peft import PeftModel
        vla = UniVLAHFModel.from_vla(vla, UniVLAHFConfig())
        vla = PeftModel.from_pretrained(vla, cfg.adapter_checkpoint, is_trainable=False)
        vla.eval()
    return vla

def get_image_resize_size(cfg):
    """
    Gets image resize size for a model class.
    If `resize_size` is an int, then the resized image will be a square.
    Else, the image will be a rectangle.
    """
    if cfg.model_family == "openvla" or cfg.model_family == "qwen2":
        resize_size = 224
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return resize_size


def get_action(cfg, model, obs, task_label, processor=None):
    """Queries the model to get an action."""
    if cfg.model_family == "openvla":
        action = get_vla_action(
            model, processor, cfg.pretrained_checkpoint, obs, task_label, cfg.unnorm_key, center_crop=cfg.center_crop, 
        )
        assert action.shape == (ACTION_DIM,)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return action


def get_latent_action(cfg, model, obs, task_label, processor=None, hist_action=''):
    """Queries the model to get an action."""
    latent_action = get_vla_latent_action(
        model, processor, cfg.pretrained_checkpoint, obs, task_label, cfg.unnorm_key, center_crop=cfg.center_crop, hist_action=hist_action,
    )

    return latent_action


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action
