"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""

from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import Metrics, VLAMetrics
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, PaddedCollatorForLanguageModeling
from prismatic.vla.action_tokenizer import ActionTokenizer

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

from typing import List, Tuple, Type, Optional, Iterable
import inspect
import torch
from torch import nn

# Optional: import common HF decoder layers for quick matches
try:
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
except Exception:
    LlamaDecoderLayer = None
try:
    from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
except Exception:
    MistralDecoderLayer = None
try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
except Exception:
    Qwen2DecoderLayer = None
try:
    from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
except Exception:
    GPTNeoXLayer = None

KNOWN_DECODER_LAYERS: Tuple[Optional[Type[nn.Module]], ...] = (
    LlamaDecoderLayer, MistralDecoderLayer, Qwen2DecoderLayer, GPTNeoXLayer
)

def infer_all_and_trainable_module_keys(model: nn.Module) -> Tuple[List[str], List[str]]:
    """
    Returns:
      all_module_keys:      top-level module names (prefix before first '.')
      trainable_module_keys:subset of above that have any requires_grad params
    """
    all_param_names = [n for n, _ in model.named_parameters()]
    if not all_param_names:
        return [], []

    # top-level prefixes (vision_backbone, projector, language_model, etc.)
    prefixes = sorted(set(n.split('.', 1)[0] for n in all_param_names))
    trainable_prefixes = set()

    for n, p in model.named_parameters():
        if p.requires_grad:
            trainable_prefixes.add(n.split('.', 1)[0])

    return prefixes, sorted(trainable_prefixes)

def _iter_modules(m: nn.Module) -> Iterable[nn.Module]:
    yield m
    for c in m.children():
        yield from _iter_modules(c)

def infer_llm_module(model: nn.Module) -> nn.Module:
    """
    Try common names first; otherwise pick a large submodule with many Linear layers.
    """
    for name in ("llm_backbone", "language_model", "text_model", "model"):
        if hasattr(model, name) and isinstance(getattr(model, name), nn.Module):
            sub = getattr(model, name)
            # Prefer the HF causal LM wrapper if present (e.g., LlamaForCausalLM.model)
            if hasattr(sub, "model") and isinstance(getattr(sub, "model"), nn.Module):
                return getattr(sub, "model")
            return sub

    # Heuristic fallback: pick the submodule with most Linear params (LLMs are linear-heavy)
    candidates = []
    for mod in _iter_modules(model):
        lin_params = sum(p.numel() for n, p in mod.named_parameters(recurse=False)
                         if isinstance(mod, nn.Module) and any(isinstance(l, nn.Linear) for l in mod.modules()))
        candidates.append((lin_params, mod))
    return max(candidates, key=lambda x: x[0])[1] if candidates else model

def infer_llm_transformer_layer_cls(model: nn.Module) -> Optional[Type[nn.Module]]:
    """
    Find the decoder layer class in a HF AutoClass model (LlamaDecoderLayer, etc.).
    """
    llm = infer_llm_module(model)

    # 1) Fast path: known classes
    for cls in KNOWN_DECODER_LAYERS:
        if cls is None:
            continue
        for m in llm.modules():
            if isinstance(m, cls):
                return cls

    # 2) Generic path: look for a class name that smells like a decoder/transformer layer
    #    and contains both attention and mlp submodules.
    for m in llm.modules():
        clsname = m.__class__.__name__.lower()
        if "decoder" in clsname or "layer" in clsname:
            has_attn = any("attn" in type(c).__name__.lower() or "attention" in type(c).__name__.lower()
                           for c in m.modules())
            has_mlp  = any("mlp"  in type(c).__name__.lower() or "ffn"       in type(c).__name__.lower()
                           for c in m.modules())
            # avoid returning the top-level LlamaModel etc.
            if has_attn and has_mlp and m is not llm:
                return m.__class__

    # 3) Couldn’t infer — return None and let callers use name-based policies instead of class checks.
    return None


# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    def __init__(
        self,
        vlm: PrismaticVLM,
        device_id: int,
        stage: str,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        **_: str,
    ) -> None:
        self.vlm, self.device_id, self.stage = vlm, device_id, stage

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        # self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        # self.llm_transformer_layer_cls = self.vlm.llm_backbone.transformer_layer_cls
        self.llm_transformer_layer_cls = infer_llm_transformer_layer_cls(self.vlm)

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None: ...

    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None: ...

    @abstractmethod
    def clip_grad_norm(self) -> None: ...

    def run_training(
        self,
        dataset: Dataset,
        collator: PaddedCollatorForLanguageModeling,
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
    ) -> None:
        """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`"""
        if "finetune" in stage and batch_construction_strategy == "split-modality":
            # Instantiate the split-modality sampler; if you want to extend with other batch construction schemes,
            #   (e.g., grouping by length) =>> can easily add them here!
            modality_lengths = dataset.get_modality_lengths()
            sampler = SplitModalitySampler(
                dataset,
                modality_lengths,
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )

        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        dataloader = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
        )

        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                self.vlm.train()
                sampler.set_epoch(epoch)

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dataloader):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            multimodal_indices=batch["multimodal_indices"],
                        )
                        loss = output.loss

                    # Commit Loss (Prior to Gradient Accumulation Normalization)
                    metrics.commit(loss=loss)

                    # Normalize Loss to account for Gradient Accumulation --> Backward!
                    # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                    #             because in general, each batch has a *different number of masked out tokens* (because
                    #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                    #
                    #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                    #             the "correct" implementation, without adding extra complexity.
                    #
                    # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                    #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                    #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                    #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)

                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                        # Push Metrics
                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                        status = metrics.push()

                        # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                        if self.max_steps is not None and metrics.global_step >= self.max_steps:
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                            if dist.is_initialized():
                                dist.barrier()

                            return

                        # Update Progress Bar
                        progress.update()
                        progress.set_description(status)

            # Save checkpoint at end each epoch (if `self.max_steps` is None)
            if self.max_steps is None:
                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                if dist.is_initialized():
                    dist.barrier()

    # === VLA Training ===

    def run_vla_training(
        self,
        vla_dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        action_tokenizer: ActionTokenizer,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        save_full_model: bool = True,
    ) -> None:
        """Run the VLA training loop for the given `dataset` and `collator`; log losses, action metrics to `metrics`."""
        assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"
        assert self.grad_accumulation_steps == 1, "VLA training does not support gradient accumulation!"

        # Create a DataLoader =>> Set `num_workers` to 0; RLDS loader handles parallelism!
        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(self.epochs * len(dataloader)) if self.max_steps is None else self.max_steps,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vlm.train()

            # Zero Gradients (just in case)
            self.optimizer.zero_grad()

            # [Contract] DataLoader wraps RLDS Loader (`.as_numpy_iterator() =>> implicit `.repeat()`)
            #   => This means looping over the DataLoader is basically "infinite" (so no outer loop over epochs).
            #      Slightly breaks default PyTorch semantics, which is why we adaptively compute `epoch` below.
            for batch in dataloader:
                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                with torch.autocast(
                    "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
                ):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    output: CausalLMOutputWithPast = self.vlm(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],
                    )
                    loss = output.loss

                # Commit Loss =>> Backward!
                metrics.commit(loss=loss)
                loss.backward()

                # # === Compute Action Token Accuracy & L1 Loss ===

                # # To compute action token accuracy, we need to identify the locations of the action tokens
                # # in both `output.logits` and `batch["labels"]`. We know that when "right" padding, we
                # # insert `self.vlm.vision_backbone.num_patches` at index 1.
                # #
                # # Computing `action_prediction_accuracy` is then pretty straightforward:
                # #   1) Extract "aligned" predictions & labels
                # #   2) Compute boolean "mask" where "labels > 2" (where 2 is ID for `EOS_TOKEN`)
                # #           => If masking out EOS, then it's just "labels != -100 (IGNORE_INDEX)
                # #   3) Compute masked accuracy as `(preds == logits) & mask` --> sum/divide by # unmasked!
                # action_preds = output.logits[:, self.vlm.vision_backbone.num_patches : -1].argmax(dim=2)
                # action_gt = batch["labels"][:, 1:].to(action_preds.device)
                # # mask = action_gt > action_tokenizer.action_token_begin_idx

                # # Mask out non-special tokens
                # mask = action_gt > 32000        

                # # Compute Accuracy
                # correct_preds = (action_preds == action_gt) & mask
                # action_accuracy = correct_preds.sum().float() / mask.sum().float()

                # # Compute L1 Loss on Predicted (Continuous) Actions
                # continuous_actions_pred = torch.tensor(
                #     action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                # )
                # continuous_actions_gt = torch.tensor(
                #     action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                # )
                # action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
                
                # # Commit Metrics
                # metrics.commit(action_accuracy=action_accuracy, l1_loss=action_l1_loss, update_step_time=True)

                # # Compute metrics per dataset --> only on rank_zero since we don't log them on other workers anyways
                # if overwatch.is_rank_zero():
                #     datasets = set(batch["dataset_names"])
                #     if len(datasets) > 1:
                #         for ds in datasets:
                #             ds_mask = torch.tensor([elem == ds for elem in batch["dataset_names"]])
                #             action_accuracy_ds = correct_preds[ds_mask].sum().float() / mask[ds_mask].sum().float()
                #             continuous_actions_pred_ds = torch.tensor(
                #                 action_tokenizer.decode_token_ids_to_actions(
                #                     action_preds[ds_mask][mask[ds_mask]].cpu().numpy()
                #                 )
                #             )
                #             continuous_actions_gt_ds = torch.tensor(
                #                 action_tokenizer.decode_token_ids_to_actions(
                #                     action_gt[ds_mask][mask[ds_mask]].cpu().numpy()
                #                 )
                #             )
                #             action_l1_loss_ds = torch.nn.functional.l1_loss(
                #                 continuous_actions_pred_ds, continuous_actions_gt_ds
                #             )

                #             metrics.commit_for_dataset(
                #                 dataset_name=ds.decode(), action_accuracy=action_accuracy_ds, l1_loss=action_l1_loss_ds
                #             )

                # === Gradient Step ===

                # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality assumptions
                self.clip_grad_norm()

                # Optimizer & LR Scheduler Step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Compute epoch value using number of completed gradient steps
                epoch = (metrics.global_step + 1) // (len(vla_dataset) // self.global_batch_size)

                # Push Metrics
                metrics.commit(global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                status = metrics.push()

                # Check for Save Interval or Max Steps & Save Checkpoint
                if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or (
                    (metrics.global_step % save_interval) == 0
                ):
                    self.save_checkpoint(
                        metrics.run_dir, metrics.global_step, epoch, loss.item(), only_trainable=not save_full_model
                    )
                    if dist.is_initialized():
                        dist.barrier()

                    if terminate:
                        return

                # Update Progress Bar
                progress.update()
                progress.set_description(status)


    # === VLA Latent Action Training ===
    def to_device(self, batch, device):
        if isinstance(batch, torch.Tensor):
            return batch.to(device, non_blocking=True)
        if isinstance(batch, dict):
            return {k: self.to_device(v, device) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            return type(batch)(self.to_device(x, device) for x in batch)
        return batch
    
    def run_latent_action_training(
        self,
        vla_dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        action_tokenizer: ActionTokenizer,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        save_full_model: bool = True,
    ) -> None:
        """Run the VLA training loop for the given `dataset` and `collator`; log losses, action metrics to `metrics`."""
        assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"
        assert self.grad_accumulation_steps == 1, "VLA training does not support gradient accumulation!"

        # Create a DataLoader =>> Set `num_workers` to 0; RLDS loader handles parallelism!
        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )

        global_bsz = self.per_device_batch_size * dist.get_world_size() if dist.is_initialized() else self.per_device_batch_size
        steps_per_epoch = (len(vla_dataset) + global_bsz - 1) // global_bsz  # ceil
        self.max_steps = self.epochs * steps_per_epoch
        print(f"We git {len(vla_dataset)} training examples")
        print(f"VLA Training: Global Batch Size = {global_bsz}, Steps per Epoch = {steps_per_epoch}")
        print(f"VLA Training: Total Steps = {self.max_steps} over {self.epochs} epochs")

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(self.epochs * len(dataloader)) if self.max_steps is None else self.max_steps,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vlm.train()

            # Zero Gradients (just in case)
            self.optimizer.zero_grad()

            # [Contract] DataLoader wraps RLDS Loader (`.as_numpy_iterator() =>> implicit `.repeat()`)
            #   => This means looping over the DataLoader is basically "infinite" (so no outer loop over epochs).
            #      Slightly breaks default PyTorch semantics, which is why we adaptively compute `epoch` below.
            for batch in dataloader:
                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                with torch.autocast(
                    "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
                ):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    # check vlm and batch device
                    batch = self.to_device(batch, next(self.vlm.parameters()).device)
                    output: CausalLMOutputWithPast = self.vlm(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],
                    )
                    loss = output.loss

                # Commit Loss =>> Backward!
                metrics.commit(loss=loss)
                loss.backward()

                # === Compute Action Token Accuracy & L1 Loss ===

                # To compute action token accuracy, we need to identify the locations of the action tokens
                # in both `output.logits` and `batch["labels"]`. We know that when "right" padding, we
                # insert `self.vlm.vision_backbone.num_patches` at index 1.
                #
                # Computing `action_prediction_accuracy` is then pretty straightforward:
                #   1) Extract "aligned" predictions & labels
                #   2) Compute boolean "mask" where "labels > 2" (where 2 is ID for `EOS_TOKEN`)
                #           => If masking out EOS, then it's just "labels != -100 (IGNORE_INDEX)
                #   3) Compute masked accuracy as `(preds == logits) & mask` --> sum/divide by # unmasked!
                action_preds = output.logits[:, self.vlm.vision_backbone.featurizer.patch_embed.num_patches : -1].argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                # Mask out non-special tokens
                mask = action_gt > 32000      



                # Compute Accuracy
                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()

                # Compute L1 Loss on Predicted (Continuous) Actions
                # continuous_actions_pred = torch.tensor(
                #     action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                # )
                # continuous_actions_gt = torch.tensor(
                #     action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                # )
                # action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
                
                # l1 loss omitted for latent action 
                action_l1_loss = torch.tensor(0.)
                # Commit Metrics
                metrics.commit(action_accuracy=action_accuracy, l1_loss=action_l1_loss, update_step_time=True)

                # Compute metrics per dataset --> only on rank_zero since we don't log them on other workers anyways
                if overwatch.is_rank_zero():
                    datasets = set(batch["dataset_names"])
                    if len(datasets) > 1:
                        for ds in datasets:
                            ds_mask = torch.tensor([elem == ds for elem in batch["dataset_names"]])
                            action_accuracy_ds = correct_preds[ds_mask].sum().float() / mask[ds_mask].sum().float()
                            # continuous_actions_pred_ds = torch.tensor(
                            #     action_tokenizer.decode_token_ids_to_actions(
                            #         action_preds[ds_mask][mask[ds_mask]].cpu().numpy()
                            #     )
                            # )
                            # continuous_actions_gt_ds = torch.tensor(
                            #     action_tokenizer.decode_token_ids_to_actions(
                            #         action_gt[ds_mask][mask[ds_mask]].cpu().numpy()
                            #     )
                            # )
                            # action_l1_loss_ds = torch.nn.functional.l1_loss(
                            #     continuous_actions_pred_ds, continuous_actions_gt_ds
                            # )
                            action_l1_loss_ds = torch.tensor(0.)
                            metrics.commit_for_dataset(
                                dataset_name=ds.decode(), action_accuracy=action_accuracy_ds, l1_loss=action_l1_loss_ds
                            )

                # === Gradient Step ===

                # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality assumptions
                self.clip_grad_norm()

                # Optimizer & LR Scheduler Step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Compute epoch value using number of completed gradient steps
                epoch = (metrics.global_step + 1) // (len(vla_dataset) // self.global_batch_size)

                # Push Metrics
                metrics.commit(global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                status = metrics.push()

                # Check for Save Interval or Max Steps & Save Checkpoint
                if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or (
                    (metrics.global_step % save_interval) == 0
                ):
                    # self.save_checkpoint(
                    #     metrics.run_dir, metrics.global_step, epoch, loss.item(), only_trainable=not save_full_model
                    # )
                    if dist.is_initialized() and dist.get_rank() == 0:
                        self.vlm.save_pretrained(metrics.run_dir / f"checkpoint-step-{metrics.global_step}", save_function=torch.save)
                        self.optimizer.state_dict()
                        torch.save(self.optimizer.state_dict(), os.path.join(metrics.run_dir / f"checkpoint-step-{metrics.global_step}", "optimizer.pt"))
                    
                    if dist.is_initialized():
                        dist.barrier()

                    if terminate:
                        return

                # Update Progress Bar
                progress.update()
                progress.set_description(status)

 