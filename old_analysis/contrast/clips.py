from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
	from transformers import CLIPModel, CLIPProcessor  # type: ignore

	_HAS_HF = True
except Exception:
	CLIPModel = None  # type: ignore
	CLIPProcessor = None  # type: ignore
	_HAS_HF = False


@dataclass
class CLIPBatch:
	"""Convenience container for a CLIP-style batch.

	- images: (B, 3, H, W) float/uint8; if uint8 should be [0,255]
	- texts: list[str] of length B
	"""

	images: torch.Tensor
	texts: Any


@dataclass
class CLIPOutput:
	image_embeds: torch.Tensor  # (B, D)
	text_embeds: torch.Tensor  # (B, D)
	logits_per_image: torch.Tensor  # (B, B)
	logits_per_text: torch.Tensor  # (B, B)
	logit_scale: torch.Tensor  # scalar


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
	return x / (x.norm(dim=dim, keepdim=True) + eps)


def clip_logits(
	image_embeds: torch.Tensor,
	text_embeds: torch.Tensor,
	logit_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Compute CLIP similarity logits.

	Args:
		image_embeds: (B, D)
		text_embeds: (B, D)
		logit_scale: scalar tensor; CLIP uses exp(logit_scale)
	"""

	image_embeds = _l2norm(image_embeds)
	text_embeds = _l2norm(text_embeds)
	scale = logit_scale.exp()
	logits_per_image = scale * image_embeds @ text_embeds.t()
	logits_per_text = logits_per_image.t()
	return logits_per_image, logits_per_text


class CLIPContrastiveLoss(nn.Module):
	"""CLIP-style symmetric InfoNCE loss.

	Default target is identity matching: target[i] == i.
	"""

	def __init__(self, label_smoothing: float = 0.0, reduction: str = "mean"):
		super().__init__()
		self.label_smoothing = float(label_smoothing)
		if reduction not in {"mean", "sum", "none"}:
			raise ValueError(f"Invalid reduction: {reduction}")
		self.reduction = reduction

	def forward(
		self,
		logits_per_image: torch.Tensor,
		logits_per_text: Optional[torch.Tensor] = None,
		*,
		targets: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		"""Compute symmetric loss.

		Args:
			logits_per_image: (B, B)
			logits_per_text: optional (B, B), if None uses transpose
			targets: optional (B,) long; if None uses arange(B)
		"""

		if logits_per_text is None:
			logits_per_text = logits_per_image.t()
		if targets is None:
			targets = torch.arange(logits_per_image.size(0), device=logits_per_image.device)

		loss_i = F.cross_entropy(
			logits_per_image,
			targets,
			label_smoothing=self.label_smoothing,
			reduction=self.reduction,
		)
		loss_t = F.cross_entropy(
			logits_per_text,
			targets,
			label_smoothing=self.label_smoothing,
			reduction=self.reduction,
		)

		if self.reduction == "none":
			return 0.5 * (loss_i + loss_t)
		return 0.5 * (loss_i + loss_t)


class HFCLIPEncoder(nn.Module):
	"""HuggingFace transformers CLIP wrapper for contrastive learning.

	- Provides `encode_image`, `encode_text`, and `forward` returning CLIPOutput.
	- Uses `CLIPProcessor` to build model inputs.
	"""

	def __init__(
		self,
		model_name: str = "openai/clip-vit-base-patch32",
		*,
		freeze_backbone: bool = False,
		device: Optional[torch.device] = None,
	):
		super().__init__()
		if not _HAS_HF:
			raise ImportError(
				"transformers not installed; please `pip install transformers` (and pillow)."
			)

		self.model_name = model_name
		self.device = device
		self.model = CLIPModel.from_pretrained(model_name)
		self.processor = CLIPProcessor.from_pretrained(model_name)

		if freeze_backbone:
			for p in self.model.parameters():
				p.requires_grad_(False)

	@property
	def logit_scale(self) -> torch.Tensor:
		return self.model.logit_scale

	def to(self, *args, **kwargs):  # type: ignore
		super().to(*args, **kwargs)
		self.model.to(*args, **kwargs)
		return self

	@torch.no_grad()
	def preprocess(self, images=None, texts=None, **kwargs) -> Dict[str, torch.Tensor]:
		"""Build model inputs. Returns tensors on module device if set."""

		inputs = self.processor(images=images, text=texts, return_tensors="pt", padding=True, **kwargs)
		if self.device is not None:
			inputs = {k: v.to(self.device) for k, v in inputs.items()}
		return inputs

	def encode_image(self, *, pixel_values: torch.Tensor) -> torch.Tensor:
		feats = self.model.get_image_features(pixel_values=pixel_values)
		return _l2norm(feats)

	def encode_text(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
		feats = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
		return _l2norm(feats)

	def forward(
		self,
		*,
		images=None,
		texts=None,
		inputs: Optional[Dict[str, torch.Tensor]] = None,
	) -> CLIPOutput:
		if inputs is None:
			inputs = self.preprocess(images=images, texts=texts)

		pixel_values = inputs.get("pixel_values")
		input_ids = inputs.get("input_ids")
		attention_mask = inputs.get("attention_mask")
		if pixel_values is None or input_ids is None or attention_mask is None:
			raise ValueError("inputs must include pixel_values, input_ids, attention_mask")

		image_embeds = self.encode_image(pixel_values=pixel_values)
		text_embeds = self.encode_text(input_ids=input_ids, attention_mask=attention_mask)
		logits_per_image, logits_per_text = clip_logits(image_embeds, text_embeds, self.logit_scale)
		return CLIPOutput(
			image_embeds=image_embeds,
			text_embeds=text_embeds,
			logits_per_image=logits_per_image,
			logits_per_text=logits_per_text,
			logit_scale=self.logit_scale,
		)


def smoke_test_hf_clip(device: Optional[str] = None) -> None:
	"""Minimal forward+loss smoke test.

	Note: requires internet on first run to download weights.
	"""

	dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
	m = HFCLIPEncoder(device=dev).to(dev)
	m.train(False)

	# Create dummy images (random) and texts.
	# CLIPProcessor expects PIL images or tensors; it supports torch tensors shaped (3,H,W).
	images = torch.rand(2, 3, 224, 224)
	texts = ["a photo of a cat", "a photo of a dog"]

	out = m(images=list(images), texts=texts)
	loss_fn = CLIPContrastiveLoss(label_smoothing=0.0)
	loss = loss_fn(out.logits_per_image, out.logits_per_text)
	assert out.logits_per_image.shape == (2, 2)
	assert torch.isfinite(loss).all()


__all__ = [
	"CLIPBatch",
	"CLIPOutput",
	"clip_logits",
	"CLIPContrastiveLoss",
	"HFCLIPEncoder",
	"smoke_test_hf_clip",
]
