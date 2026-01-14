#%%
import os
from pathlib import Path
import torch
import sys
sys.path.append("/fs/scratch/PAS2099/danielf/geometry_grounded_latents/UniVLA")
from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel

# --- Config ---
model_name = "uni_lam"
lam_path = "/fs/scratch/PAS2099/danielf/geometry_grounded_latents/UniVLA/checkpoints/univla-latent-action-model/lam-stage-2.ckpt"

# Model hyperparameters
cfg = {
    "in_dim": 3,
    "model_dim": 768,
    "latent_dim": 128,
    "num_latents": 16,
    "patch_size": 14,
    "enc_blocks": 12,
    "dec_blocks": 12,
    "num_heads": 12,
    "dropout": 0.0,
}

# --- Initialize and load model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_action_model = ControllableDINOLatentActionModel(**cfg)

# Load checkpoint
ckpt = torch.load(lam_path, map_location='cpu')['state_dict']
ckpt = {k.replace("lam.", ""): v for k, v in ckpt.items()}

latent_action_model.load_state_dict(ckpt, strict=True)
latent_action_model = latent_action_model.to(device).eval()

print("Latent Action Model loaded successfully on", device)

#%%
import sys
sys.path.append("/fs/scratch/PAS2099/danielf/geometry_grounded_latents/UniVLA/latent_action_model")
from genie.dataset import LightningOpenX

batch_size = 256

ds = LightningOpenX(
    data_root='/fs/scratch/PAS2099/danielf/geometry_grounded_latents/data',
    data_mix='bridge',
    batch_size=batch_size,
    shuffle_buffer_size=4500,
    num_workers=16,
)

ds.setup('fit')

loader = ds.train_dataloader()

#%%
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

save_path = f"/fs/scratch/PAS2099/danielf/geometry_grounded_latents/LAPA/laq/laq_model/top_bins/{model_name}"

bins = dict()  # codebook index -> count
model = latent_action_model

for batch_idx, batch in enumerate(tqdm(loader), start=1):
    bs = batch['videos'].shape[0]
    # batch['videos']: [B, T, C, H, W]
    # resize to 224x224
    videos_resized = torch.nn.functional.interpolate(batch['videos'].view(-1, *batch['videos'].shape[2:]), size=(224, 224), mode='bilinear', align_corners=False)
    videos_resized = videos_resized.view(batch['videos'].shape[0], batch['videos'].shape[1], *videos_resized.shape[1:])

    with torch.no_grad():
        indices = model.vq_encode(videos_resized.to(device))['indices']
    indices_flat = indices.view(indices.size(0), -1)  # [B, L]
    for b in range(bs):
        code = indices_flat[b].cpu().numpy()  # [L,]
        # convert code to string for dict key
        code_str = ','.join(map(str, code))
        if code_str not in bins:
            bins[code_str] = 0
        bins[code_str] += 1

    # every 10 batches, write (override) the stats file
    if (batch_idx % 10) == 0:
        import json
        Path(save_path).mkdir(parents=True, exist_ok=True)
        fp = os.path.join(save_path, f'temp.json')
        with open(fp, 'w') as f:
            json.dump(bins, f)
        print(f"Saved action frequency bins at batch {batch_idx} (unique={len(bins)})")

# final save to ensure latest stats persisted
import json
Path(save_path).mkdir(parents=True, exist_ok=True)
with open(os.path.join(save_path, f'{model_name}_action_freq.json'), 'w') as f:
    json.dump(bins, f)
print("Saved final action frequency bins.")

# %%

model_name = "uni_lam"
saved_path = f"/fs/scratch/PAS2099/danielf/geometry_grounded_latents/LAPA/laq/laq_model/top_bins/{model_name}/{model_name}_action_freq.json"
import json
with open(saved_path, 'r') as f:
    bins = json.load(f)

# sort by frequency
sorted_bins = sorted(bins.items(), key=lambda x: x[1], reverse=True)
# plot a histogram
import matplotlib.pyplot as plt
freqs = [v for k,v in sorted_bins]
plt.figure(figsize=(10,5))
plt.bar(range(len(freqs)), freqs, width=1.1)
plt.yscale('log')
plt.xlabel('Action Code Index (sorted)')
plt.ylabel('Frequency (log scale)')
plt.title(f'Action Code Frequency Distribution ({model_name} bridge)')

plt.show()

#%%
