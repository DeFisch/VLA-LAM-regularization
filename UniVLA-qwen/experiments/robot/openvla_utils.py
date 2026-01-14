"""Utils for evaluating the OpenVLA policy."""

from locale import str
import json
import os
from pyparsing import Optional
import time

import numpy as np
import peft
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

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


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with Flash-Attention Enabled")

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation=None,
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Move model to device.
    # Note: `.to()` is not supported for 8-bit or 4-bit bitsandbytes models, but the model will
    #       already be set to the right devices and casted to the correct dtype upon loading.
    # We handle DDP evaluation in CALVIN with accelerator instead.
    if not cfg.load_in_8bit and not cfg.load_in_4bit and ('libero' in cfg.task_suite_name or 'r2r' in cfg.task_suite_name):
        vla = vla.to(DEVICE)

    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Process inputs.
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    # Get action.
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=True, top_p=0.75)
    return action


def predict_latent_action(
    vla, input_ids = None, unnorm_key = None, **kwargs: str
    ) -> np.ndarray:
    """Thin wrapper around .generate() that decodes predicted actions and unnormalizes them."""
    # # If the special empty token ('') does not already appear after the colon (':') token in the prompt
    # # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
    # if not torch.all(input_ids[:, -1] == 29871):
    #     input_ids = torch.cat(
    #         (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
    #     )

    # Run VLA inference
    output = super(vla.__class__, vla).generate(input_ids, min_new_tokens=4, max_new_tokens=4, return_dict_in_generate=True, output_hidden_states=True, **kwargs)
    generated_ids = output.sequences

    output = vla(input_ids=generated_ids, output_hidden_states=True, pixel_values=kwargs.get('pixel_values', None), use_cache=False)

    # last_hidden_states = [hidden_states[-1] for hidden_states in output.hidden_states]
    # latent_tokens = torch.cat(last_hidden_states, dim=1)#[:, :-1]
    latent_tokens = output.hidden_states[-1]
    P = vla.vision_backbone.dino_featurizer.patch_embed.num_patches
    visual_embed = latent_tokens[:, 1:1+P, :].to(torch.float)
    # visual_embed = latent_tokens[:, :P, :].to(torch.float)
    latent_tokens = torch.cat([latent_tokens[:, :1, :].to(torch.float), 
                               latent_tokens[:, 1+P:, :].to(torch.float)], 
                               dim=1)
    # latent_tokens = latent_tokens[:, P:, :].to(torch.float)

    # print(generated_ids)
    latent_mask = (generated_ids >= 151665) & (generated_ids <= 151680) # 16 action tokens
    latent_mask = latent_mask[:, 1:]
    # print(latent_mask[0])
    # latent_action = latent_tokens[:, latent_mask[0], :]
    latent_action = latent_tokens[:, -4:]
    generated_ids = generated_ids[:, 1:][:, latent_mask[0]]
    generated_ids = generated_ids[:, -4:]

    return visual_embed, latent_action, generated_ids

def get_vla_latent_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False, hist_action=''):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    import peft
    if type(vla) == peft.peft_model.PeftModel:
        vla = vla.vla

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    
    if len(hist_action) > 0:
        prompt = f"In: What action should the robot take to {task_label.lower()}? History action {hist_action}\nOut:"


    # print(prompt)
    if hasattr(vla.vision_backbone, 'get_image_transform'):
        # if the model is QwenVLA
        # Process inputs.
        image_transform = vla.vision_backbone.get_image_transform()
        image = image_transform(image) # image: dict with 'dino' and 'siglip' keys
        # to device
        image = {k: v.unsqueeze(0).to(vla.device) for k, v in image.items()}
        tokenizer = vla.llm_backbone.get_tokenizer()
        text_inputs = tokenizer(prompt, add_special_tokens=True)
        input_ids = text_inputs.input_ids
        # append 220 to each input ids tensor
        input_ids = torch.cat([torch.tensor(input_ids), torch.tensor([220])], dim=0)
        labels = list(input_ids)
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        
        visual_embed, latent_tokens, generated_tokens = predict_latent_action(
            vla,
            input_ids=input_ids.unsqueeze(0).to(vla.device),
            pixel_values=image,
            do_sample=True,
            temperature=0.75,
            top_p=0.9,
        )
        generated_tokens = tokenizer.decode(generated_tokens[0].cpu().numpy(), skip_special_tokens=False, clean_up_tokenization_spaces=True)
        # print("Generated tokens:", generated_tokens)
        return visual_embed, latent_tokens, generated_tokens
    else:
        inputs = processor(prompt, image).to(vla.device, dtype=torch.bfloat16)

        # Get latent action.
        action = vla.predict_latent_action(**inputs, unnorm_key=unnorm_key, do_sample=True, temperature=0.75, top_p = 0.9)

        return action