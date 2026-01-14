import torch
import numpy as np
import cv2

import sys
sys.path.append("/fs/scratch/PAS2099/danielf/geometry_grounded_latents/mast3r")  # adjust to your mast3r path
# If your env needs the dust3r/mast3r path setup, keep your existing import side-effects:
import mast3r.utils.path_to_dust3r  # noqa
from mast3r.model import AsymmetricMASt3R
from dust3r.model import AsymmetricCroCo3DStereo  # fallback

def extract_mast3r_latent(
    img1,
    img2,
    device: torch.device,
    weights_path="/fs/scratch/PAS2099/danielf/geometry_grounded_latents/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    image_size=224,          # e.g., 224; if None, keep input size
    normalize=False,          # set True if your checkpoint expects ImageNet norm
    last_k=1,                 # use last K decoder layers (1 = just last)
    concat_views=True,        # if True, concat view1/view2 along tokens -> (B, 2*S, D)
):
    """
    Inputs (img1, img2) can be:
      - torch.Tensor of shape (B, C, H, W) or (C, H, W) or (H, W, C)
      - np.ndarray of shape (B, H, W, C) or (H, W, C)
      with RGB in uint8 [0..255] or float [0..1].

    Returns:
      latent: torch.FloatTensor
        If concat_views=True:  shape (B, 2*S, D)
        else:                  tuple(latent1, latent2), each (B, S, D)
    """
    # -------------------------
    # 1) Load model (MASt3R -> fallback to CroCo3D)
    # -------------------------
    model = None
    try:
      model = AsymmetricMASt3R.from_pretrained(weights_path).to(device).eval()
    except Exception:
      try:
        model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(device).eval()
      except Exception as e:
        raise RuntimeError(f"Could not load weights: {weights_path}\n{e}")

    # -------------------------
    # 2) Convert inputs to torch (B, C, H, W), float32 in [0,1], on device
    # -------------------------
    def to_tensor_bchw(x):
        # numpy -> torch
        if isinstance(x, np.ndarray):
            if x.ndim == 3:           # (H, W, C)
                x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            elif x.ndim == 4:         # (B, H, W, C)
                x = torch.from_numpy(x).permute(0, 3, 1, 2)            # (B, C, H, W)
            else:
                raise ValueError(f"Unsupported numpy shape: {x.shape}")
        elif isinstance(x, torch.Tensor):
            if x.ndim == 3:           # (C, H, W) or (H, W, C)
                if x.shape[0] in (1,3):   # assume (C, H, W)
                    x = x.unsqueeze(0)    # (1, C, H, W)
                else:                      # assume (H, W, C)
                    x = x.permute(2,0,1).unsqueeze(0)
            elif x.ndim == 4:
                pass  # (B, C, H, W)
            else:
                raise ValueError(f"Unsupported tensor shape: {tuple(x.shape)}")
        else:
            raise TypeError("img must be np.ndarray or torch.Tensor")

        # ensure channel-first
        assert x.shape[1] in (1,3), f"Expected channels=1 or 3, got {x.shape}"
        x = x.float()
        # scale to [0,1] if looks like uint8
        if x.max() > 1.0:
            x = x / 255.0
        return x

    t1 = to_tensor_bchw(img1)
    t2 = to_tensor_bchw(img2)

    # (optional) resize to a square size that matches your checkpoint (e.g., 224 or 256)
    def maybe_resize(x, size):
        if size is None:
            return x
        return torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    t1 = maybe_resize(t1, image_size)
    t2 = maybe_resize(t2, image_size)

    # optional normalization (ImageNet)
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406], device=t1.device)[None, :, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225], device=t1.device)[None, :, None, None]
        t1 = (t1 - mean) / std
        t2 = (t2 - mean) / std

    t1 = t1.to(device, non_blocking=True)
    t2 = t2.to(device, non_blocking=True)

    B, C, H, W = t1.shape
    assert t2.shape[:2] == (B, C), "Batch and channel must match between views"
    assert t2.shape[2:] == (H, W), "Spatial size must match between views"

    # -------------------------
    # 3) Build MASt3R view dicts (img, true_shape, instance)
    # -------------------------
    true_shape = torch.tensor([H, W], device=device).repeat(B, 1)
    instances_1 = [f"pair_{i}_v1" for i in range(B)]
    instances_2 = [f"pair_{i}_v2" for i in range(B)]

    view1 = {"img": t1, "true_shape": true_shape, "instance": instances_1}
    view2 = {"img": t2, "true_shape": true_shape, "instance": instances_2}

    # -------------------------
    # 4) Forward to get decoder tokens
    # -------------------------
    with torch.no_grad():
        dec1, dec2 = model.forward_latents(view1, view2)  # each is a list length L (e.g., 13)

    # -------------------------
    # 5) Take final (or last-K averaged) latents and optionally concat views
    # -------------------------
    def last_k_avg(dec_list, k):
        if k == 1:
            return dec_list[-1].float()            # (B, S, D)
        k = min(k, len(dec_list))
        return torch.mean(torch.stack([d.float() for d in dec_list[-k:]], dim=0), dim=0)  # (B, S, D)

    z1 = last_k_avg(dec1, last_k)   # (B, S, D)
    z2 = last_k_avg(dec2, last_k)   # (B, S, D)

    if concat_views:
        # concat along the token dimension -> (B, 2*S, D)
        latent = torch.cat([z1, z2], dim=1)
        return latent
    else:
        return z1, z2
