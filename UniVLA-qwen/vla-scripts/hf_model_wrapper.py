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