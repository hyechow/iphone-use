"""GUIClip backend for page similarity: CLIP ViT-B/32 fine-tuned on GUI screenshots."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from PIL import Image


class GUIClipBackend:
    """CLIP ViT-B/32 fine-tuned on GUI data for screenshot similarity.

    Model is lazily loaded on first ``similarity()`` call (~577 MB).
    """

    def __init__(self, model_name: str = "Jl-wei/guiclip-vit-base-patch32"):
        self._model_name = model_name
        self._model: Any = None
        self._processor: Any = None

    # -- lazy loading --------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import CLIPModel, CLIPProcessor

        print(f"[GUIClip] Loading model {self._model_name} ...")
        self._processor = CLIPProcessor.from_pretrained(self._model_name, local_files_only=True)
        self._model = CLIPModel.from_pretrained(self._model_name, local_files_only=True)
        self._model.eval()
        print("[GUIClip] Model loaded")

    # -- SimilarityBackend interface -----------------------------------------

    def similarity(self, png_a: bytes, png_b: bytes) -> float:
        import torch
        import torch.nn.functional as F

        self._ensure_loaded()

        img_a = Image.open(io.BytesIO(png_a)).convert("RGB")
        img_b = Image.open(io.BytesIO(png_b)).convert("RGB")

        with torch.no_grad():
            inputs_a = self._processor(images=img_a, return_tensors="pt")
            inputs_b = self._processor(images=img_b, return_tensors="pt")

            pixel_a = {k: v for k, v in inputs_a.items() if k.startswith("pixel")}
            pixel_b = {k: v for k, v in inputs_b.items() if k.startswith("pixel")}

            emb_a = self._model.visual_projection(
                self._model.vision_model(**pixel_a).pooler_output
            )
            emb_b = self._model.visual_projection(
                self._model.vision_model(**pixel_b).pooler_output
            )

            emb_a = F.normalize(emb_a, p=2, dim=1)
            emb_b = F.normalize(emb_b, p=2, dim=1)

            return float(F.cosine_similarity(emb_a, emb_b).item())
