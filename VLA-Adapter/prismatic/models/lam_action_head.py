"""
lam_action_head.py

LAM Prediction Head - Uses VLA-Adapter's MLPResNet architecture to predict LAM codes directly.
Output: 4 tokens × 16 classes (instead of continuous 7-dim actions)
"""

import torch
import torch.nn as nn


# LAM constants
NUM_LAM_TOKENS = 4  # LAM produces 4 tokens
NUM_LAM_CLASSES = 16  # Each token is 0-15


class MLPResNetBlock(nn.Module):
    """Simple MLP residual block."""
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x, h_t=None, h_a=None, p=None):
        residual = x
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x + residual


class MLPResNetBlock_Pro(nn.Module):
    """MLP residual block with cross-attention to VLM hidden states."""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(dim)

        # Cross-attention to task (vision) hidden states
        self.cross_attn_t = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_attn_t_norm = nn.LayerNorm(dim)

        # Cross-attention to action hidden states
        self.cross_attn_a = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_attn_a_norm = nn.LayerNorm(dim)

    def forward(self, x, h_t=None, h_a=None, p=None):
        residual = x

        # Cross-attention to task hidden states
        if h_t is not None:
            x_norm = self.cross_attn_t_norm(x)
            attn_out, _ = self.cross_attn_t(x_norm, h_t, h_t)
            x = x + attn_out

        # Cross-attention to action hidden states
        if h_a is not None:
            x_norm = self.cross_attn_a_norm(x)
            attn_out, _ = self.cross_attn_a(x_norm, h_a, h_a)
            x = x + attn_out

        # MLP
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x + residual


class LAMPredictionHead(nn.Module):
    """
    LAM Prediction Head using VLA-Adapter's MLPResNet architecture.

    Predicts LAM codes directly: 4 tokens × 16 classes
    Uses the same cross-attention mechanism as L1RegressionActionHead.
    """

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        num_task_tokens=512,
        num_blocks=24,
        use_pro_version=True,  # Use cross-attention version by default
    ):
        super().__init__()
        self.num_task_tokens = num_task_tokens
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # Input projection (from VLM hidden dim to internal hidden dim)
        # Input: (batch, seq_len, input_dim * action_dim_slots)
        # We use a single slot for LAM prediction
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

        # MLP ResNet blocks with cross-attention
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            if use_pro_version:
                self.mlp_resnet_blocks.append(MLPResNetBlock_Pro(dim=hidden_dim))
            else:
                self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))

        # Output layer norm
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # Output projection to LAM logits
        # Output: 4 tokens × 16 classes = 64 values
        self.fc_out = nn.Linear(hidden_dim, NUM_LAM_TOKENS * NUM_LAM_CLASSES)

    def forward(self, hidden_states, h_t=None, h_a=None, proprio=None):
        """
        Forward pass for LAM prediction.

        Args:
            hidden_states: VLM hidden states (batch, num_layers, seq_len, hidden_dim)
            h_t: Task hidden states for cross-attention (batch, num_layers, num_patches, hidden_dim)
            h_a: Action hidden states for cross-attention (batch, num_layers, num_action_tokens, hidden_dim)
            proprio: Not used (kept for API compatibility)

        Returns:
            logits: (batch, 4, 16) - logits for each LAM token position
        """
        batch_size = hidden_states.shape[0]

        # Use the last layer's hidden states, mean-pooled across sequence
        # hidden_states: (batch, num_layers, seq_len, hidden_dim)
        x = hidden_states[:, -1, :, :]  # (batch, seq_len, hidden_dim)
        x = x.mean(dim=1, keepdim=True)  # (batch, 1, hidden_dim) - pool to single vector

        # Input projection
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)

        # Process through MLP ResNet blocks with cross-attention
        for i, block in enumerate(self.mlp_resnet_blocks):
            # Use corresponding layer's hidden states for cross-attention
            layer_idx = min(i + 1, h_t.shape[1] - 1) if h_t is not None else 0
            h_t_layer = h_t[:, layer_idx, :, :] if h_t is not None else None
            h_a_layer = h_a[:, layer_idx, :, :] if h_a is not None else None

            x = block(x, h_t=h_t_layer, h_a=h_a_layer, p=proprio)

        # Output projection
        x = self.layer_norm2(x)
        logits = self.fc_out(x)  # (batch, 1, 64)

        # Reshape to (batch, 4, 16)
        logits = logits.squeeze(1)  # (batch, 64)
        logits = logits.view(batch_size, NUM_LAM_TOKENS, NUM_LAM_CLASSES)

        return logits


class LAMPredictionHeadV2(nn.Module):
    """
    Simpler LAM Prediction Head - direct classification from VLM hidden states.

    Uses attention pooling over VLM hidden states, then MLP to predict LAM codes.
    """

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=1024,
        num_heads=8,
        num_layers=2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project VLM hidden states
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
        )

        # Learnable query tokens for each LAM position
        self.queries = nn.Parameter(torch.randn(NUM_LAM_TOKENS, hidden_dim) * 0.02)

        # Transformer decoder for cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, NUM_LAM_CLASSES),
        )

    def forward(self, hidden_states, h_t=None, h_a=None, proprio=None):
        """
        Forward pass for LAM prediction.

        Args:
            hidden_states: VLM hidden states (batch, num_layers, seq_len, hidden_dim)

        Returns:
            logits: (batch, 4, 16) - logits for each LAM token position
        """
        batch_size = hidden_states.shape[0]

        # Use last layer hidden states
        x = hidden_states[:, -1, :, :]  # (batch, seq_len, hidden_dim)

        # Project to internal dimension
        memory = self.input_proj(x)  # (batch, seq_len, hidden_dim)

        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, 4, hidden_dim)

        # Cross-attention: queries attend to VLM hidden states
        output = self.decoder(queries, memory)  # (batch, 4, hidden_dim)

        # Project to class logits
        logits = self.output_proj(output)  # (batch, 4, 16)

        return logits


class LAMPredictionHeadV3(nn.Module):
    """
    Autoregressive LAM Prediction Head with teacher forcing.

    Uses cross-attention to VLM hidden states and predicts tokens autoregressively:
    P(z1|ctx) * P(z2|ctx,z1) * P(z3|ctx,z1,z2) * P(z4|ctx,z1,z2,z3)

    Key features:
    1. Cross-attention allows selective attention to relevant features
    2. Autoregressive structure captures token dependencies
    3. Teacher forcing during training for faster convergence
    """

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=1024,
        num_heads=8,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Project VLM hidden states to internal dimension
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
        )

        # Learnable query embeddings for each LAM token position
        self.query_embeddings = nn.Parameter(torch.randn(NUM_LAM_TOKENS, hidden_dim) * 0.02)

        # Token embedding for autoregressive conditioning (embed previous token predictions)
        self.token_embedding = nn.Embedding(NUM_LAM_CLASSES + 1, hidden_dim)  # +1 for start token

        # Positional embedding for the 4 LAM token positions
        self.pos_embedding = nn.Parameter(torch.randn(NUM_LAM_TOKENS, hidden_dim) * 0.02)

        # Transformer decoder layers with cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection to class logits
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, NUM_LAM_CLASSES),
        )

        # Causal mask for autoregressive decoding
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(NUM_LAM_TOKENS, NUM_LAM_TOKENS), diagonal=1).bool()
        )

    def forward(self, hidden_states, targets=None, h_t=None, h_a=None, proprio=None):
        """
        Forward pass with cross-attention and teacher forcing.

        Args:
            hidden_states: VLM hidden states (batch, num_layers, seq_len, hidden_dim)
            targets: Ground truth LAM tokens (batch, 4) for teacher forcing.
                     If None, uses autoregressive generation (inference mode).

        Returns:
            logits: (batch, 4, 16) - logits for each token position
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device

        # Use last layer hidden states
        x = hidden_states[:, -1, :, :]  # (batch, seq_len, hidden_dim)

        # Project to internal dimension
        memory = self.input_proj(x)  # (batch, seq_len, hidden_dim)

        if targets is not None:
            # Teacher forcing: use ground truth tokens for conditioning
            # Shift targets right and prepend start token (use NUM_LAM_CLASSES as start token idx)
            start_tokens = torch.full((batch_size, 1), NUM_LAM_CLASSES, dtype=torch.long, device=device)
            shifted_targets = torch.cat([start_tokens, targets[:, :-1]], dim=1)  # (batch, 4)

            # Embed the shifted tokens
            token_embeds = self.token_embedding(shifted_targets)  # (batch, 4, hidden_dim)

            # Add query embeddings and positional embeddings
            queries = self.query_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, 4, hidden_dim)
            queries = queries + token_embeds + self.pos_embedding.unsqueeze(0)

            # Apply transformer decoder with causal masking
            output = self.decoder(
                queries,
                memory,
                tgt_mask=self.causal_mask,
            )  # (batch, 4, hidden_dim)

            # Project to logits
            logits = self.output_proj(output)  # (batch, 4, NUM_LAM_CLASSES)

        else:
            # Autoregressive generation (inference)
            logits_list = []
            prev_token = torch.full((batch_size,), NUM_LAM_CLASSES, dtype=torch.long, device=device)  # Start token

            for i in range(NUM_LAM_TOKENS):
                # Embed previous token
                token_embed = self.token_embedding(prev_token)  # (batch, hidden_dim)

                # Query for current position
                query = self.query_embeddings[i:i+1].unsqueeze(0).expand(batch_size, -1, -1)  # (batch, 1, hidden_dim)
                query = query + token_embed.unsqueeze(1) + self.pos_embedding[i:i+1].unsqueeze(0)

                # Cross-attention to memory
                output = self.decoder(query, memory)  # (batch, 1, hidden_dim)

                # Project to logits
                step_logits = self.output_proj(output.squeeze(1))  # (batch, NUM_LAM_CLASSES)
                logits_list.append(step_logits)

                # Get predicted token for next step
                prev_token = step_logits.argmax(dim=-1)  # (batch,)

            logits = torch.stack(logits_list, dim=1)  # (batch, 4, NUM_LAM_CLASSES)

        return logits
