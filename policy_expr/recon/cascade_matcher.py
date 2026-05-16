"""Cascade page matcher: shared GUIClip + LLM + bge-small-zh models."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

FINGERPRINT_PROMPT = """为这个 App 截图生成"页面模板描述"，目标是：同一类型页面（内容不同、结构相同）的描述尽量接近；不同类型页面的描述尽量有区别。

严格按以下格式输出三行，不要添加任何其他内容：
用途：<这类页面让用户做什么，动宾短语，对该类型所有实例都成立>
内容区：<主内容区的数据组织形式，描述结构而非具体数据，如"可滚动的会话条目列表"/"按时间排列的气泡对话流"/"单一对象的字段详情卡">
交互方式：<用户与页面的主要交互，如"滚动浏览并点击进入详情"/"输入文字并发送消息"/"切换开关配置选项"/"全屏查看并左右翻页">

规则（违反则描述无效）：
- 假设页面换了不同用户的数据，你的描述仍然成立——不能描述任何具体内容（特定商品、消息正文、金额、用户名）
- 不提状态栏、顶部标题/返回按钮、底部固定导航栏——这些每页都有，不具备区分价值
- 全部中文，每行不超过 25 字"""


@dataclass
class PageEmbedding:
    """Visual + optional text embedding for a screenshot."""

    visual: np.ndarray          # GUIClip embedding, shape (512,), L2-normalised
    text: np.ndarray | None = None  # bge-small-zh embedding, shape (512,), lazy


class CascadeMatcher:
    """Holds GUIClip and bge-small-zh models; generates and compares page embeddings.

    All three sub-models (GUIClip, bge, LLM client) are loaded lazily on first use.
    Intended to be created once and shared across PageDedup and PageComparator.
    """

    def __init__(
        self,
        guiclip_model: str = "Jl-wei/guiclip-vit-base-patch32",
        embed_model: str = "BAAI/bge-small-zh",
        llm_config_key: str = "fingerprint",
    ):
        self._guiclip_name = guiclip_model
        self._embed_name = embed_model
        self._llm_config_key = llm_config_key
        self._clip_model: Any = None
        self._clip_proc: Any = None
        self._embed: Any = None
        self._llm: Any = None
        self._llm_model: str = ""

    # ── lazy loading ──────────────────────────────────────────────────────────

    def _load_guiclip(self) -> None:
        if self._clip_model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor
        self._clip_proc = CLIPProcessor.from_pretrained(
            self._guiclip_name, local_files_only=True
        )
        self._clip_model = CLIPModel.from_pretrained(
            self._guiclip_name, local_files_only=True
        )
        self._clip_model.eval()

    def _load_embed(self) -> None:
        if self._embed is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._embed = SentenceTransformer(self._embed_name, local_files_only=True)

    def _load_llm(self) -> None:
        if self._llm is not None:
            return
        from openai import OpenAI
        from policy_expr.config import resolve_llm_config
        cfg = resolve_llm_config(self._llm_config_key)
        self._llm = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        self._llm_model = cfg.model

    # ── internal ops ──────────────────────────────────────────────────────────

    def _compute_visual(self, png: bytes) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        self._load_guiclip()
        img = Image.open(io.BytesIO(png)).convert("RGB")
        with torch.no_grad():
            inputs = self._clip_proc(images=img, return_tensors="pt")
            pixel = {k: v for k, v in inputs.items() if k.startswith("pixel")}
            emb = self._clip_model.visual_projection(
                self._clip_model.vision_model(**pixel).pooler_output
            )
            emb = F.normalize(emb, p=2, dim=1)
        return emb.squeeze(0).numpy()

    def _compute_text(self, fingerprint: str) -> np.ndarray:
        self._load_embed()
        return self._embed.encode(fingerprint, normalize_embeddings=True)

    def _generate_fingerprint(self, png: bytes) -> str:
        self._load_llm()
        b64 = base64.b64encode(png).decode()
        resp = self._llm.chat.completions.create(
            model=self._llm_model,
            max_tokens=200,
            extra_body={"enable_thinking": False},
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": FINGERPRINT_PROMPT},
            ]}],
        )
        return resp.choices[0].message.content.strip()

    # ── public embedding API ──────────────────────────────────────────────────

    def embed_visual(self, png: bytes) -> PageEmbedding:
        """Compute visual-only embedding (fast, no LLM call)."""
        return PageEmbedding(visual=self._compute_visual(png))

    def embed_full(self, png: bytes) -> PageEmbedding:
        """Compute visual + text embeddings (calls LLM for fingerprint)."""
        visual = self._compute_visual(png)
        fingerprint = self._generate_fingerprint(png)
        text = self._compute_text(fingerprint)
        return PageEmbedding(visual=visual, text=text)

    def fill_text(self, emb: PageEmbedding, png: bytes) -> None:
        """Lazily generate and attach text embedding if not already present."""
        if emb.text is None:
            fingerprint = self._generate_fingerprint(png)
            emb.text = self._compute_text(fingerprint)

    # ── similarity ────────────────────────────────────────────────────────────

    def visual_sim(self, a: PageEmbedding, b: PageEmbedding) -> float:
        return float(np.dot(a.visual, b.visual))

    def text_sim(self, a: PageEmbedding, b: PageEmbedding) -> float:
        if a.text is None or b.text is None:
            raise ValueError("text embeddings not populated; call fill_text first")
        return float(np.dot(a.text, b.text))
