"""
action_heads.py

Implementations of various action heads, which serve as alternatives to VLM sequential token prediction.
"""

import math
import torch
import torch.nn as nn
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX, NUM_TOKENS



def learnable_random_perturbations(seq_len, dim, device, dtype):
    """Generate random perturbations. Uses regular tensor, not nn.Parameter, to avoid memory leaks."""
    with torch.no_grad():
        random_perturbations = torch.randn(seq_len, dim, device=device, dtype=dtype) * 0.02
    return random_perturbations



class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression.

    Optionally includes LAM latent prediction head for auxiliary discrete action prediction.
    """
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_task_tokens=512,
        use_pro_version=False,
        use_lam=False,  # If True, enable LAM latent prediction (4 tokens x 16 classes)
        lam_tap_block=24,  # Which MLP block's output to use for LAM prediction (1-24)
        lam_decoder_type="transformer",  # "simple" or "transformer"
        lam_use_vlm_hidden=False,  # If True, use VLM hidden states instead of MLPResNet output
    ):
        super().__init__()
        self.num_task_tokens = num_task_tokens
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_lam = use_lam
        self.lam_tap_block = lam_tap_block
        self.lam_decoder_type = lam_decoder_type
        self.lam_use_vlm_hidden = lam_use_vlm_hidden
        self.model = MLPResNet(
            num_blocks=24,
            input_dim=input_dim*ACTION_DIM,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            use_pro_version=use_pro_version,
            use_lam=use_lam,
            lam_tap_block=lam_tap_block,
            lam_decoder_type=lam_decoder_type,
            lam_use_vlm_hidden=lam_use_vlm_hidden,
        )

    def predict_action(
            self,
            actions_hidden_states,
            proprio=None,
            proprio_projector=None,
            phase="Inference",
            return_lam_logits=False,
            lam_targets=None,
            ):
        """
        Predict continuous actions and optionally LAM latent logits.

        Args:
            actions_hidden_states: Hidden states from VLM (batch, num_layers, seq_len, hidden_dim)
            proprio: Proprioceptive state
            proprio_projector: Module to project proprio to hidden dim
            phase: "Training" or "Inference"
            return_lam_logits: If True and LAM head exists, also return LAM logits
            lam_targets: Ground truth LAM tokens (B, 4) for teacher forcing during training.
                         If None, uses autoregressive generation (inference mode).

        Returns:
            If return_lam_logits=False: actions (batch, chunk_len, action_dim)
            If return_lam_logits=True: (actions, lam_logits) where lam_logits is (batch, 4, 16)
                - 4 categorical distributions modeled autoregressively
        """
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device

        # Handle proprio features (optional)
        if proprio is not None and proprio_projector is not None:
            proprio = proprio.reshape(batch_size, -1).to(torch.bfloat16)  # (bsz, proprio_dim)
            proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
            proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)
        else:
            proprio_features = None

        task_hidden_states = actions_hidden_states[:, :, :self.num_task_tokens, :]
        actions_hidden_states = actions_hidden_states[:, :, self.num_task_tokens:, :]

        cond_actions_hidden_states = torch.zeros(
            (batch_size, self.action_dim * NUM_ACTIONS_CHUNK, self.hidden_dim),
            device=device, dtype=actions_hidden_states.dtype
        ).detach()

        rearranged_actions_hidden_states = cond_actions_hidden_states.reshape(
            batch_size, NUM_ACTIONS_CHUNK, -1
        )  # (batch, chunk_len, action_dim * hidden_dim)

        if phase == "Training":
            batch_size, seq_len, dim = rearranged_actions_hidden_states.shape
            random_perturbations = learnable_random_perturbations(seq_len, dim, device=rearranged_actions_hidden_states.device, dtype=rearranged_actions_hidden_states.dtype)
            rearranged_actions_hidden_states = (rearranged_actions_hidden_states + random_perturbations)  # (1, seq_len, dim)

        result = self.model(
            rearranged_actions_hidden_states,
            h_a=actions_hidden_states,
            p=proprio_features,
            h_t=task_hidden_states,
            return_lam_logits=return_lam_logits and self.use_lam,
            lam_targets=lam_targets,
        )

        return result
    

class SimpleLAMDecoder(nn.Module):
    """
    Simple 4-head independent LAM decoder.

    Predicts all 4 LAM tokens independently (no autoregression).
    Each token has its own output head: P(z1|x), P(z2|x), P(z3|x), P(z4|x)

    This is a simpler baseline to test if hidden states contain LAM-relevant info.
    """

    NUM_TOKENS = 4  # LAM produces 4 tokens
    NUM_CLASSES = 16  # Each token is 0-15

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Shared context projection
        self.context_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 4 independent output heads - one for each LAM token
        self.token_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, self.NUM_CLASSES),
            )
            for _ in range(self.NUM_TOKENS)
        ])

    def forward(self, context, targets=None):
        """
        Forward pass for independent LAM token prediction.

        Args:
            context: Pooled representation from MLP block output (B, hidden_dim)
            targets: Not used (kept for API compatibility)

        Returns:
            logits: (B, 4, 16) - logits for each token position
        """
        # Project context
        projected = self.context_proj(context)  # (B, hidden_dim)

        # Predict each token independently with its own head
        all_logits = []
        for head in self.token_heads:
            logits = head(projected)  # (B, 16)
            all_logits.append(logits)

        # Stack to (B, 4, 16)
        return torch.stack(all_logits, dim=1)


class TransformerLAMDecoder(nn.Module):
    """
    Transformer-based LAM decoder with cross-attention.

    Uses cross-attention to attend to VLM hidden states instead of mean pooling.
    Predicts LAM tokens autoregressively: P(z1|ctx) * P(z2|ctx,z1) * P(z3|ctx,z1,z2) * P(z4|ctx,z1,z2,z3)

    Key improvements over SimpleLAMDecoder:
    1. Cross-attention allows selective attention to relevant features
    2. Autoregressive structure captures token dependencies
    3. Learnable query embeddings for each LAM token position
    """

    NUM_TOKENS = 4  # LAM produces 4 tokens
    NUM_CLASSES = 16  # Each token is 0-15

    def __init__(self, hidden_dim, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Learnable query embeddings for each LAM token position
        self.query_embeddings = nn.Parameter(torch.randn(self.NUM_TOKENS, hidden_dim) * 0.02)

        # Token embedding for autoregressive conditioning (embed previous token predictions)
        self.token_embedding = nn.Embedding(self.NUM_CLASSES + 1, hidden_dim)  # +1 for start token

        # Positional embedding for the 4 LAM token positions
        self.pos_embedding = nn.Parameter(torch.randn(self.NUM_TOKENS, hidden_dim) * 0.02)

        # Input projection for context (VLM hidden states)
        self.context_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Transformer decoder layers with cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection to class logits
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.NUM_CLASSES),
        )

        # Causal mask for autoregressive decoding
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(self.NUM_TOKENS, self.NUM_TOKENS), diagonal=1).bool()
        )

    def forward(self, context, targets=None):
        """
        Forward pass with cross-attention to context.

        Args:
            context: VLM hidden states (B, seq_len, hidden_dim) - NOT mean-pooled!
            targets: Ground truth LAM tokens (B, 4) for teacher forcing.
                     If None, uses autoregressive generation (slower).

        Returns:
            logits: (B, 4, 16) - logits for each token position
        """
        B = context.shape[0]
        device = context.device

        # Project context
        memory = self.context_proj(context)  # (B, seq_len, hidden_dim)

        if targets is not None:
            # Teacher forcing: use ground truth tokens for conditioning
            # Shift targets right and prepend start token (use NUM_CLASSES as start token idx)
            start_tokens = torch.full((B, 1), self.NUM_CLASSES, dtype=torch.long, device=device)
            shifted_targets = torch.cat([start_tokens, targets[:, :-1]], dim=1)  # (B, 4)

            # Embed the shifted tokens
            token_embeds = self.token_embedding(shifted_targets)  # (B, 4, hidden_dim)

            # Add query embeddings and positional embeddings
            queries = self.query_embeddings.unsqueeze(0).expand(B, -1, -1)  # (B, 4, hidden_dim)
            queries = queries + token_embeds + self.pos_embedding.unsqueeze(0)

            # Apply transformer decoder with causal masking
            output = self.transformer_decoder(
                queries,
                memory,
                tgt_mask=self.causal_mask,
            )  # (B, 4, hidden_dim)

            # Project to logits
            logits = self.output_proj(output)  # (B, 4, NUM_CLASSES)

        else:
            # Autoregressive generation (inference)
            logits_list = []
            prev_token = torch.full((B,), self.NUM_CLASSES, dtype=torch.long, device=device)  # Start token

            for i in range(self.NUM_TOKENS):
                # Embed previous token
                token_embed = self.token_embedding(prev_token)  # (B, hidden_dim)

                # Query for current position
                query = self.query_embeddings[i:i+1].unsqueeze(0).expand(B, -1, -1)  # (B, 1, hidden_dim)
                query = query + token_embed.unsqueeze(1) + self.pos_embedding[i:i+1].unsqueeze(0)

                # Cross-attention to memory
                # For single query, no causal mask needed
                output = self.transformer_decoder(query, memory)  # (B, 1, hidden_dim)

                # Project to logits
                step_logits = self.output_proj(output.squeeze(1))  # (B, NUM_CLASSES)
                logits_list.append(step_logits)

                # Get predicted token for next step
                prev_token = step_logits.argmax(dim=-1)  # (B,)

            logits = torch.stack(logits_list, dim=1)  # (B, 4, NUM_CLASSES)

        return logits


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""

    # LAM code constants
    NUM_LAM_TOKENS = 4  # LAM produces 4 tokens
    NUM_LAM_CLASSES = 16  # Each token is 0-15

    def __init__(
            self,
            num_blocks,
            input_dim,
            hidden_dim,
            output_dim,
            use_pro_version=False,
            use_lam=False,  # If True, add LAM latent prediction head
            lam_tap_block=24,  # Which block's output to use for LAM prediction (1-24)
            lam_decoder_type="transformer",  # "simple" or "transformer"
            lam_use_vlm_hidden=False,  # If True, use VLM hidden states directly instead of MLPResNet output
            ):

        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_lam = use_lam
        self.num_blocks = num_blocks
        self.lam_tap_block = lam_tap_block
        self.lam_decoder_type = lam_decoder_type
        self.lam_use_vlm_hidden = lam_use_vlm_hidden
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()

        for _ in range(num_blocks):
            if use_pro_version:
                self.mlp_resnet_blocks.append(MLPResNetBlock_Pro(dim=hidden_dim))
            else:
                self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))

        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # LAM latent prediction head (optional)
        # Two decoder options:
        # - "simple": Independent 4-head MLP decoder (mean-pools hidden states)
        # - "transformer": Cross-attention decoder (uses full sequence)
        if use_lam:
            if lam_decoder_type == "transformer":
                self.lam_decoder = TransformerLAMDecoder(
                    hidden_dim=hidden_dim,
                    num_heads=8,
                    num_layers=2,
                    dropout=0.1,
                )
            else:
                self.lam_decoder = SimpleLAMDecoder(
                    hidden_dim=hidden_dim,
                    dropout=0.1,
                )
        else:
            self.lam_decoder = None


    def forward(self, x, h_a=None, h_t=None, p=None, return_lam_logits=False, lam_targets=None):
        """
        Forward pass through MLPResNet.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            h_a: Action hidden states from VLM layers
            h_t: Task hidden states from VLM layers
            p: Proprio features
            return_lam_logits: If True and lam_decoder exists, also return LAM logits
            lam_targets: Ground truth LAM tokens (B, 4) for teacher forcing during training.
                         If None during LAM prediction, uses autoregressive generation.

        Returns:
            If return_lam_logits=False: action predictions (batch_size, seq_len, output_dim)
            If return_lam_logits=True: (actions, lam_logits) where lam_logits is (batch_size, 4, 16)
                - 4 categorical distributions modeled autoregressively
        """
        # x: (batch_size, seq_len, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, seq_len, input_dim)
        x = self.fc1(x)  # shape: (batch_size, seq_len, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, seq_len, hidden_dim)

        # For LAM, we'll tap hidden states at a specific block
        lam_repr = None

        for i, block in enumerate(self.mlp_resnet_blocks):
            x = block(x, h_t=h_t[:,i+1,:], h_a=h_a[:,i+1,:], p=p)  # shape: (batch_size, seq_len, hidden_dim)

            # Tap hidden states at the specified block for LAM prediction
            if return_lam_logits and (i + 1) == self.lam_tap_block:
                lam_repr = x.clone()

        # Shared representation after all 24 blocks
        shared_repr = self.layer_norm2(x)  # shape: (batch_size, seq_len, hidden_dim)

        # Continuous action output
        actions = self.fc2(shared_repr)  # shape: (batch_size, seq_len, output_dim)

        if return_lam_logits and self.lam_decoder is not None:
            # Choose representation source for LAM prediction
            if self.lam_use_vlm_hidden:
                # Use VLM hidden states directly (richer visual information)
                # h_t contains vision patch features from VLM layer at lam_tap_block
                # h_t shape: (batch_size, num_layers, num_patches, hidden_dim)
                # Use the layer corresponding to lam_tap_block
                layer_idx = min(self.lam_tap_block, h_t.shape[1] - 1)
                lam_repr = h_t[:, layer_idx, :, :]  # (batch_size, num_patches, hidden_dim)
            else:
                # Use MLPResNet hidden states (tapped at lam_tap_block)
                if lam_repr is None:
                    # Fallback: use final representation if tap block wasn't reached
                    lam_repr = x

            # Different handling based on decoder type
            if self.lam_decoder_type == "transformer":
                # TransformerLAMDecoder uses full sequence with cross-attention
                # No pooling - pass (batch_size, seq_len, hidden_dim)
                lam_logits = self.lam_decoder(
                    lam_repr,
                    targets=lam_targets,
                )  # (batch_size, 4, 16)
            else:
                # SimpleLAMDecoder uses mean-pooled representation
                pooled_repr = lam_repr.mean(dim=1)  # (batch_size, hidden_dim)
                lam_logits = self.lam_decoder(
                    pooled_repr,
                    targets=lam_targets,
                )  # (batch_size, 4, 16)

            return actions, lam_logits

        return actions   



def apply_rope(q, k, cos, sin):
    """
    RoPE:
    q, k: (B, H, T, D)   # D must be an even number
    cos/sin: (T, D)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)


    def rotate_half(x):
        # Swap even and odd dimensions and flip the signs
        x1 = x[..., ::2]   # Even subdimension
        x2 = x[..., 1::2]  # odd subdimension

        return torch.stack((-x2, x1), dim=-1).reshape_as(x)


    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    return q_rot, k_rot



class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        """
        dim = head_dim
        """
        super().__init__()
        assert dim % 2 == 0, "RoPE head_dim must be an even number"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (T, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)            # (T, dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)



class MLPResNetBlock(nn.Module):
    """
    One residual MLP block with cross-attention conditioning.

    This block applies multi-head attention over:
      - token features (self-attention),
      - task-related hidden states (h_t),
      - action/proprioception-related hidden states (h_a, p).
    The outputs are combined via a gating mechanism, projected back to the
    hidden dimension, and passed through a small feedforward sub-network with
    residual connection.

    Args:
        dim (int): Dimensionality of the hidden features. Must be divisible by num_heads.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
        h_t (torch.Tensor, optional): Task-related hidden states of shape
                                      (batch_size, K, hidden_dim).
        h_a (torch.Tensor, optional): Action-related hidden states of shape
                                      (batch_size, 1, hidden_dim).
        p (torch.Tensor, optional): Additional conditioning features
                                    (e.g., proprioception), shape (batch_size, 1, hidden_dim).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim).
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Main feedforward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        self.num_heads = 8
        self.head_dim = dim // self.num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.gating_factor = nn.Parameter(torch.zeros(1))



    def forward(self, x, h_t=None, h_a=None, p=None):
        """
        x: (batch_size, seq_len, hidden_dim)
        h, t, p: (batch_size, 1, hidden_dim) or None
        """

        g = self.gating_factor
        ratio_g = nn.Tanh()(g)

        conditions = []
        if h_a is not None:
            conditions.append(h_a)
        if p is not None:
            conditions.append(p)

        h = torch.cat(conditions, dim=1)  # (batch_size, cond_len, hidden_dim)

        B = x.size(0)
        T = x.size(1)
        C = x.size(2)
        K_t = h.size(1)
        K = h_t.size(1)

        task_k = h
        task_v = h

        adapter_k = h_t
        adapter_v = h_t

        q_1 = self.q_proj(x) # (B, T, C)
        k_tokens = self.k_proj(x)             # (B, T, C)
        v_tokens = self.v_proj(x)             # (B, T, C)
        k_task = self.k_proj(task_k)    # (B, K, C)
        v_task = self.v_proj(task_v)    # (B, K, C)

        k_adapter = self.k_proj(adapter_k)    # (B, K, C)
        v_adapter = self.v_proj(adapter_v)    # (B, K, C)

        # (B, seq_len, C) -> (B, num_heads, seq_len, head_dim)
        q_1 = q_1.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        k_tokens = k_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_tokens = v_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_task = k_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)
        v_task = v_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)

        k_adapter = k_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v_adapter = v_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores_tokens = torch.matmul(q_1, k_tokens.transpose(-2, -1)) # (B, H, T, T)
        attn_scores_task = torch.matmul(q_1, k_task.transpose(-2, -1)) * 1 # (B, H, T, K)
        attn_scores_adapter = torch.matmul(q_1, k_adapter.transpose(-2, -1)) * ratio_g # (B, H, T, K)

        attn_scores = torch.cat([attn_scores_tokens, attn_scores_task, attn_scores_adapter], dim=-1) # (B, H, T, T+K)
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1) # (B, H, T, T+K)

        v_combined = torch.cat([v_tokens, v_task, v_adapter], dim=2) # (B, H, T+K, head_dim)
        output = torch.matmul(attn_weights, v_combined) # (B, H, T, head_dim)

        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        x = self.ffn(output + x) 

        return x



class MLPResNetBlock_Pro(nn.Module):
    """One MLP ResNet block with separate projections for self, adapter, task + RoPE, now with FiLM modulation."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            )

        # Q (from x only)
        self.q_proj = nn.Linear(dim, dim)

        # Self-Attention: K, V
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)

        # Adapter cross-attention: K, V
        self.k_adapter = nn.Linear(dim, dim)
        self.v_adapter = nn.Linear(dim, dim)

        # Task cross-attention: K, V
        self.k_task = nn.Linear(dim, dim)
        self.v_task = nn.Linear(dim, dim)

        self.o_proj = nn.Linear(dim, dim)

        # gating
        self.gating_factor = nn.Parameter(torch.zeros(1))

        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim)

        # ---- FiLM ----
        # FiLM is useless; to avoid conflict with chkpt, it can be kept as is for now.
        self.film_gen = nn.Sequential(
            nn.Linear(dim, dim * 2),  # output γ and β
            )


    def apply_film(self, x, gamma, beta):
        """FiLM: per-channel modulation"""
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)


    def forward(self, x, h_a=None, h_t=None, p=None):
        """
        h_a: adapter tokens
        h_t: task tokens
        p:   possible conditioning vector (for FiLM)
        """
        g = self.gating_factor
        ratio_g = torch.tanh(g)

        # concat h_a and p
        h_adapter = torch.cat((h_a, p),dim=1)

        h_task = h_t
        B, T, C = x.shape
        K_a = h_adapter.size(1) if h_a is not None else 0
        K_t = h_task.size(1) if h_task is not None else 0

        # Q
        q_1 = self.q_proj(x)

        # self tokens
        k_tokens = self.k_self(x)
        v_tokens = self.v_self(x)

        # adapter tokens
        k_adapter = self.k_adapter(h_adapter)
        v_adapter = self.v_adapter(h_adapter)

        # task tokens
        k_task = self.k_task(h_task)
        v_task = self.v_task(h_task)


        # reshape -> multi-head
        def reshape_heads(t, B, L):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)


        q_1 = reshape_heads(q_1, B, T)
        k_tokens, v_tokens = reshape_heads(k_tokens, B, T), reshape_heads(v_tokens, B, T)
        k_adapter, v_adapter = reshape_heads(k_adapter, B, K_a), reshape_heads(v_adapter, B, K_a)
        k_task, v_task = reshape_heads(k_task, B, K_t), reshape_heads(v_task, B, K_t)

        # RoPE
        cos_main, sin_main = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
        q_1, k_tokens = apply_rope(q_1, k_tokens, cos_main, sin_main)
        cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
        _, k_adapter = apply_rope(k_adapter, k_adapter, cos_a, sin_a)     
        cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
        _, k_task = apply_rope(k_task, k_task, cos_t, sin_t)

        # attention scores
        attn_scores = [torch.matmul(q_1, k_tokens.transpose(-2, -1))]
        attn_scores.append(torch.matmul(q_1, k_adapter.transpose(-2, -1)))
        attn_scores.append(torch.matmul(q_1, k_task.transpose(-2, -1)) * ratio_g)
        attn_scores = torch.cat(attn_scores, dim=-1) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # combine V
        v_list = [v_tokens,v_adapter,v_task]
        v_combined = torch.cat(v_list, dim=2)

        output = torch.matmul(attn_weights, v_combined)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        # # ---- FiLM ---- 
        # gamma_beta = self.film_gen(p)  # [B, 2C]
        # gamma, beta = gamma_beta.chunk(2, dim=-1)  # [B, C], [B, C]
        # output = self.apply_film(output, gamma, beta)

        # residual + FFN
        x = self.ffn(output + x)
        return x
