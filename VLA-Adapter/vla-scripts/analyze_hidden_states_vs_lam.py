"""
Analyze what information action hidden states contain vs what LAM codes represent.

This script investigates:
1. Can we predict LAM codes from action hidden states at different processing stages?
2. Where in the architecture is LAM-relevant information preserved/lost?
3. What's the maximum achievable LAM accuracy from action hidden states?

Probing locations:
- VLM output hidden states (before any MLP processing)
- After MLP input projection (fc1)
- After different numbers of MLP ResNet blocks (e.g., 6, 12, 18, 24)
- Final pooled representation (what LAM decoder currently receives)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoProcessor
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.vla.datasets import RLDSDataset, RLDSBatchTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.constants import NUM_TOKENS, ACTION_DIM, NUM_ACTIONS_CHUNK
from torch.utils.data import DataLoader

# LAM imports
LAM_PATH = "/home/daniel/code/lam-latent/UniVLA/lam_ckpts/stage2_libero"
sys.path.insert(0, "/home/daniel/code/lam-latent/UniVLA")
from latent_action_model.genie.st_mask_git import ControllableDINOLatentActionModel

NUM_LAM_TOKENS = 4
NUM_LAM_CLASSES = 16


class LinearProbe(nn.Module):
    """Simple linear probe to predict LAM codes from hidden states."""
    def __init__(self, input_dim, num_tokens=4, num_classes=16):
        super().__init__()
        self.num_tokens = num_tokens
        # Predict all 4 tokens jointly
        self.fc = nn.Linear(input_dim, num_tokens * num_classes)

    def forward(self, x):
        # x: (B, input_dim)
        logits = self.fc(x)  # (B, num_tokens * num_classes)
        return logits.view(-1, self.num_tokens, NUM_LAM_CLASSES)  # (B, 4, 16)


class MLPProbe(nn.Module):
    """2-layer MLP probe for more expressive prediction."""
    def __init__(self, input_dim, hidden_dim=512, num_tokens=4, num_classes=16):
        super().__init__()
        self.num_tokens = num_tokens
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_tokens * num_classes),
        )

    def forward(self, x):
        logits = self.net(x)
        return logits.view(-1, self.num_tokens, NUM_LAM_CLASSES)


def load_lam_model(device):
    """Load the LAM model for extracting ground truth codes."""
    lam_model = ControllableDINOLatentActionModel.from_pretrained(LAM_PATH)
    lam_model = lam_model.to(device).eval()
    for param in lam_model.parameters():
        param.requires_grad = False
    return lam_model


def extract_lam_codes(lam_model, initial_frame, final_frame):
    """Extract LAM codes from frame pairs."""
    with torch.no_grad():
        # Stack frames: (B, 2, C, H, W)
        frames = torch.stack([initial_frame, final_frame], dim=1)

        # Get LAM codes
        result = lam_model.vq_encode(frames)
        indices = result["indices"]  # (B, 4) - controllable indices from stage-2

    return indices


def collect_hidden_states_at_stages(action_head, multi_layer_hidden_states, proprio_features):
    """
    Manually forward through MLPResNet and collect hidden states at different stages.

    Returns dict with hidden states at:
    - 'input': Before any MLP processing (raw VLM hidden states, pooled)
    - 'fc1': After input projection
    - 'block_6': After 6 MLP blocks
    - 'block_12': After 12 MLP blocks
    - 'block_18': After 18 MLP blocks
    - 'block_24': After all 24 MLP blocks (final)
    """
    model = action_head.model
    batch_size = multi_layer_hidden_states.shape[0]
    device = multi_layer_hidden_states.device

    # Prepare input (same as in predict_action)
    task_hidden_states = multi_layer_hidden_states[:, :, :action_head.num_task_tokens, :]
    actions_hidden_states = multi_layer_hidden_states[:, :, action_head.num_task_tokens:, :]

    cond_actions_hidden_states = torch.zeros(
        (batch_size, ACTION_DIM * NUM_ACTIONS_CHUNK, action_head.hidden_dim),
        device=device, dtype=multi_layer_hidden_states.dtype
    ).detach()

    x = cond_actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)

    hidden_states = {}

    # Pool VLM hidden states for comparison
    # Use mean across layers and sequence
    vlm_pooled = multi_layer_hidden_states.mean(dim=(1, 2))  # (B, hidden_dim)
    hidden_states['vlm_pooled'] = vlm_pooled.detach()

    # Forward through fc1
    x = model.layer_norm1(x)
    x = model.fc1(x)
    x = model.relu(x)
    hidden_states['fc1'] = x.mean(dim=1).detach()  # Pool across sequence

    # Forward through blocks, collecting at checkpoints
    h_t = task_hidden_states
    h_a = actions_hidden_states
    p = proprio_features

    for i, block in enumerate(model.mlp_resnet_blocks):
        x = block(x, h_t=h_t[:, i+1, :], h_a=h_a[:, i+1, :], p=p)

        if (i + 1) in [6, 12, 18, 24]:
            hidden_states[f'block_{i+1}'] = x.mean(dim=1).detach()

    # Final layer norm
    shared_repr = model.layer_norm2(x)
    hidden_states['final'] = shared_repr.mean(dim=1).detach()

    return hidden_states


def train_probe(probe, train_features, train_targets, num_epochs=100, lr=1e-3):
    """Train a probe on collected features."""
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(train_features, train_targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    probe.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = probe(batch_x)  # (B, 4, 16)

            # Compute CE loss for each token position
            loss = 0
            for t in range(NUM_LAM_TOKENS):
                loss += F.cross_entropy(logits[:, t, :], batch_y[:, t])
            loss = loss / NUM_LAM_TOKENS

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return probe


def evaluate_probe(probe, features, targets):
    """Evaluate probe accuracy."""
    probe.eval()
    with torch.no_grad():
        logits = probe(features)  # (B, 4, 16)
        preds = logits.argmax(dim=-1)  # (B, 4)

        # Per-token accuracy
        token_correct = (preds == targets).float().mean(dim=0)  # (4,)
        avg_token_acc = token_correct.mean().item()

        # Sequence accuracy (all 4 tokens correct)
        seq_correct = (preds == targets).all(dim=1).float().mean().item()

        # Per-token CE loss
        ce_loss = 0
        for t in range(NUM_LAM_TOKENS):
            ce_loss += F.cross_entropy(logits[:, t, :], targets[:, t]).item()
        ce_loss = ce_loss / NUM_LAM_TOKENS

    return {
        'token_acc': avg_token_acc,
        'seq_acc': seq_correct,
        'ce_loss': ce_loss,
        'per_token_acc': token_correct.cpu().numpy(),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    print("Loading VLA model...")
    vla_path = "openvla/openvla-7b"
    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)

    vla = OpenVLAForActionPrediction.from_pretrained(
        vla_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    vla.eval()

    # Load action head
    print("Loading action head...")
    action_head_path = "/home/daniel/code/lam-latent/VLA-Adapter/runs/libero_goal_all_12-09_23:10/checkpoint_step_27200/action_head.pt"
    action_head = L1RegressionActionHead(
        input_dim=896,
        hidden_dim=4096,
        action_dim=7,
        num_task_tokens=729,  # 27x27 patches
        use_pro_version=True,
        use_lam=True,
    )
    action_head.load_state_dict(torch.load(action_head_path, map_location=device))
    action_head = action_head.to(device).eval()

    # Load proprio projector
    proprio_proj_path = "/home/daniel/code/lam-latent/VLA-Adapter/runs/libero_goal_all_12-09_23:10/checkpoint_step_27200/proprio_projector.pt"
    from prismatic.models.projectors import ProprioProjector
    proprio_projector = ProprioProjector(proprio_dim=8, llm_dim=896)
    proprio_projector.load_state_dict(torch.load(proprio_proj_path, map_location=device))
    proprio_projector = proprio_projector.to(device).eval()

    print("Loading LAM model...")
    lam_model = load_lam_model(device)

    # Load dataset
    print("Loading dataset...")
    dataset = RLDSDataset(
        data_root_dir=Path("/home/daniel/tensorflow_datasets"),
        data_mix="libero_goal_all",
        image_transform=RLDSBatchTransform(processor.image_processor),
        tokenizer=processor.tokenizer,
        prompt_builder_fn=lambda x: x,
        default_image_resolution=(224, 224),
        shuffle_buffer_size=10000,
        image_aug=False,
        window_size=12,  # For LAM: frames 0 and 11
        future_action_window_size=0,
        compute_lam=True,
        lam_model_path=LAM_PATH,
    )

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )

    dataloader = DataLoader(dataset, batch_size=8, collate_fn=collator, num_workers=0)

    # Collect hidden states and LAM codes
    print("\nCollecting hidden states at different stages...")
    all_hidden_states = defaultdict(list)
    all_lam_codes = []

    num_samples = 500  # Collect 500 samples for probing
    samples_collected = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, total=num_samples // 8):
            if samples_collected >= num_samples:
                break

            # Get LAM codes
            if "lam_latent" in batch and batch["lam_latent"] is not None:
                lam_codes = batch["lam_latent"].to(device)
            else:
                # Skip if no LAM codes
                continue

            # Forward through VLA
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)

            output = vla(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )

            # Extract hidden states at different layers
            batch_size = input_ids.shape[0]
            num_patches = 729  # 27x27

            # Build multi-layer hidden states (same as training)
            multi_layer_hidden_states = []
            for layer_idx in range(1, 25):  # Layers 1-24
                item = output.hidden_states[layer_idx]
                # Extract vision and action tokens
                task_latent_states = item[:, :num_patches].reshape(batch_size, 1, num_patches, -1)
                # For simplicity, use zeros for action hidden states (we care about task/vision)
                action_hs = torch.zeros(batch_size, 1, NUM_TOKENS, item.shape[-1], device=device, dtype=item.dtype)
                all_hs = torch.cat((task_latent_states, action_hs), 2)
                multi_layer_hidden_states.append(all_hs)

            multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim=1).to(torch.bfloat16)

            # Get proprio features
            proprio = batch["proprio"].to(device).reshape(batch_size, -1).to(torch.bfloat16)
            proprio_features = proprio_projector(proprio).unsqueeze(1)

            # Collect hidden states at different stages
            stage_hidden = collect_hidden_states_at_stages(
                action_head, multi_layer_hidden_states, proprio_features
            )

            for stage_name, hidden in stage_hidden.items():
                all_hidden_states[stage_name].append(hidden.float().cpu())

            all_lam_codes.append(lam_codes.cpu())
            samples_collected += batch_size

    # Concatenate all collected data
    print(f"\nCollected {samples_collected} samples")

    for stage_name in all_hidden_states:
        all_hidden_states[stage_name] = torch.cat(all_hidden_states[stage_name], dim=0)
    all_lam_codes = torch.cat(all_lam_codes, dim=0)

    print(f"Hidden state shapes: {[(k, v.shape) for k, v in all_hidden_states.items()]}")
    print(f"LAM codes shape: {all_lam_codes.shape}")

    # Split into train/test
    n_train = int(0.8 * samples_collected)

    # Train and evaluate probes at each stage
    print("\n" + "="*60)
    print("Training probes at different processing stages...")
    print("="*60)

    results = {}

    for stage_name, features in all_hidden_states.items():
        print(f"\n--- Stage: {stage_name} ---")

        train_x = features[:n_train].to(device)
        train_y = all_lam_codes[:n_train].to(device)
        test_x = features[n_train:].to(device)
        test_y = all_lam_codes[n_train:].to(device)

        input_dim = features.shape[-1]

        # Train linear probe
        linear_probe = LinearProbe(input_dim).to(device)
        train_probe(linear_probe, train_x, train_y, num_epochs=100)
        linear_results = evaluate_probe(linear_probe, test_x, test_y)

        # Train MLP probe
        mlp_probe = MLPProbe(input_dim, hidden_dim=512).to(device)
        train_probe(mlp_probe, train_x, train_y, num_epochs=100)
        mlp_results = evaluate_probe(mlp_probe, test_x, test_y)

        print(f"  Linear Probe: Token Acc={linear_results['token_acc']:.3f}, Seq Acc={linear_results['seq_acc']:.3f}, CE={linear_results['ce_loss']:.3f}")
        print(f"  MLP Probe:    Token Acc={mlp_results['token_acc']:.3f}, Seq Acc={mlp_results['seq_acc']:.3f}, CE={mlp_results['ce_loss']:.3f}")

        results[stage_name] = {
            'linear': linear_results,
            'mlp': mlp_results,
        }

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: LAM Prediction Accuracy at Different Stages")
    print("="*60)
    print(f"{'Stage':<15} {'Linear Token':<15} {'Linear Seq':<12} {'MLP Token':<12} {'MLP Seq':<10}")
    print("-"*60)

    for stage_name in ['vlm_pooled', 'fc1', 'block_6', 'block_12', 'block_18', 'block_24', 'final']:
        if stage_name in results:
            r = results[stage_name]
            print(f"{stage_name:<15} {r['linear']['token_acc']:.3f}          {r['linear']['seq_acc']:.3f}        {r['mlp']['token_acc']:.3f}        {r['mlp']['seq_acc']:.3f}")

    print("\nRandom baseline: Token Acc = 0.0625 (1/16), Seq Acc = 0.000015 (1/16^4)")

    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    # Check if accuracy decreases through blocks
    if 'fc1' in results and 'block_24' in results:
        fc1_acc = results['fc1']['mlp']['token_acc']
        final_acc = results['block_24']['mlp']['token_acc']

        if fc1_acc > final_acc + 0.05:
            print(f"⚠️  Information LOSS detected: fc1 ({fc1_acc:.3f}) -> block_24 ({final_acc:.3f})")
            print("   LAM-relevant information is being lost through MLP blocks.")
        elif final_acc > fc1_acc + 0.05:
            print(f"✓  Information GAIN detected: fc1 ({fc1_acc:.3f}) -> block_24 ({final_acc:.3f})")
            print("   MLP blocks are extracting/refining LAM-relevant information.")
        else:
            print(f"→  Information roughly preserved: fc1 ({fc1_acc:.3f}) -> block_24 ({final_acc:.3f})")


if __name__ == "__main__":
    main()
