"""Report builder: generate HTML visualization for exploration/execution logs."""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ── Action types & colors ──────────────────────────────────────

ACTION_COLORS: dict[str, tuple[int, int, int]] = {
    "tap": (220, 50, 50),
    "home": (50, 120, 220),
    "swipe": (50, 180, 50),
    "text": (160, 50, 200),
    "back": (220, 160, 0),
    "none": (128, 128, 128),
}

DEFAULT_COLOR = ACTION_COLORS["tap"]

COLOR_NAVIGATED = (34, 197, 94)    # green
COLOR_NO_CHANGE = (156, 163, 175)  # gray

DOT_RADIUS = 14
FONT_SIZE = 16


def _font(size: int = FONT_SIZE):
    try:
        return ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", size)
    except Exception:
        return ImageFont.load_default()


_REPORT_MAX_W = 640  # resize annotated images to this width before saving


def _save_report_img(src: "Image.Image | bytes", path: Path, quality: int = 75) -> None:
    """Save an image as JPEG to disk, resizing to _REPORT_MAX_W if wider."""
    if isinstance(src, bytes):
        img = Image.open(io.BytesIO(src)).convert("RGB")
    else:
        img = src.convert("RGB")
    w, h = img.size
    if w > _REPORT_MAX_W:
        img = img.resize((_REPORT_MAX_W, round(h * _REPORT_MAX_W / w)), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG", quality=quality, optimize=True)


def _load_img(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def annotate_tap(
    img: Image.Image,
    points: list[tuple[float, float, int]],
) -> Image.Image:
    """Draw numbered tap points on image. points: (x_pct, y_pct, index)."""
    img = img.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    font = _font(FONT_SIZE)

    for x_pct, y_pct, idx in points:
        cx = int(x_pct / 1000 * w)
        cy = int(y_pct / 1000 * h)
        color = DEFAULT_COLOR
        draw.ellipse(
            [cx - DOT_RADIUS, cy - DOT_RADIUS, cx + DOT_RADIUS, cy + DOT_RADIUS],
            fill=(*color, 200),
            outline=(255, 255, 255, 255),
            width=2,
        )
        text = str(idx)
        bbox = font.getbbox(text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((cx - tw // 2, cy - th // 2), text, fill=(255, 255, 255, 255), font=font)

    return img


def annotate_recon_taps(
    img: Image.Image,
    points: list[tuple[float, float, int, bool]],
) -> Image.Image:
    """Draw recon tap points: green=navigated, gray=no change. points: (x_pct, y_pct, index, navigated)."""
    img = img.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    font = _font(FONT_SIZE)

    for x_pct, y_pct, idx, navigated in points:
        cx = int(x_pct / 1000 * w)
        cy = int(y_pct / 1000 * h)
        color = COLOR_NAVIGATED if navigated else COLOR_NO_CHANGE
        draw.ellipse(
            [cx - DOT_RADIUS, cy - DOT_RADIUS, cx + DOT_RADIUS, cy + DOT_RADIUS],
            fill=(*color, 210),
            outline=(255, 255, 255, 255),
            width=2,
        )
        text = str(idx)
        bbox = font.getbbox(text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((cx - tw // 2, cy - th // 2), text, fill=(255, 255, 255, 255), font=font)

    return img


LOGICAL_W = 318  # iPhone Mirroring logical pixel width (matches executor.WIN_W)

STRATEGY_COLORS: dict[str, tuple[int, int, int]] = {
    "fixed": (99, 102, 241),   # indigo
    "YOLO":    (234, 179,  8),   # amber
    "LLM":     (59, 130, 246),   # blue
    "LLM+YOLO":(139,  92, 246),  # violet
}


def annotate_back_attempts_img(
    img: Image.Image,
    attempts: list[dict],
) -> Image.Image:
    """Draw numbered back-attempt circles: green=success, red=fail, by strategy color."""
    img = img.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    w, _ = img.size
    scale = w / LOGICAL_W
    font = _font(13)

    for i, a in enumerate(attempts, 1):
        coords = a.get("coords", [])
        if len(coords) < 2:
            continue
        cx = int(coords[0] * scale)
        cy = int(coords[1] * scale)
        success = a.get("success", False)
        strategy = a.get("strategy", "")
        color = (34, 197, 94) if success else STRATEGY_COLORS.get(strategy, (239, 68, 68))
        draw.ellipse(
            [cx - DOT_RADIUS, cy - DOT_RADIUS, cx + DOT_RADIUS, cy + DOT_RADIUS],
            fill=(*color, 220),
            outline=(255, 255, 255, 255),
            width=2,
        )
        text = str(i)
        bbox = font.getbbox(text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((cx - tw // 2, cy - th // 2), text, fill=(255, 255, 255, 255), font=font)

    return img


def img_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


# ── Runner data classes ────────────────────────────────────────

@dataclass
class ReportStep:
    label: str
    action_type: str  # tap, home, swipe, text, none
    x: float | None   # 0-1000 normalized
    y: float | None
    description: str
    annotated_before_url: str  # base64 data url with tap point drawn
    after_url: str | None      # screenshot after action
    status: str = ""  # ✓ ✗ ↩ ""
    timestamp: str = ""  # ISO timestamp


@dataclass
class ReportPage:
    title: str
    steps: list[ReportStep] = field(default_factory=list)


@dataclass
class ReportData:
    title: str
    pages: list[ReportPage] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# ── Recon data classes ─────────────────────────────────────────

@dataclass
class ReconTap:
    index: int
    label: str
    x: float
    y: float
    navigated: bool
    after_url: str | None  # screenshot after tap
    back_seq: list[dict] = field(default_factory=list)  # {"src": data_url, "subtitle": str, "success": bool}


@dataclass
class ReconFlow:
    target_page: str
    target_description: str
    flow_description: str


@dataclass
class ReconPageInfo:
    name: str            # directory name
    title: str           # from recon_result.json
    page_type: str       # list, chat, detail, etc.
    description: str
    elements_count: int
    signature: str
    annotated_url: str   # initial.png with all taps annotated
    taps: list[ReconTap] = field(default_factory=list)
    flows: list[ReconFlow] = field(default_factory=list)
    knowledge: str = ""
    error: dict | None = None  # structured error if probe was aborted
    error_annotated_url: str = ""  # failed-tap screenshot annotated with back-attempt coords


@dataclass
class AppReconData:
    app_name: str
    pages: list[ReconPageInfo] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    trace: list[dict] | None = None  # from trace.json if available


@dataclass
class NavNode:
    name: str
    page: ReconPageInfo | None  # None = unexplored (not yet visited by BFS)
    via_tap: str | None = None  # label of the tap that led to this page
    children: list["NavNode"] = field(default_factory=list)


def _build_nav_tree(pages: list[ReconPageInfo], trace: list[dict] | None = None) -> list[NavNode]:
    """Build navigation tree. Uses trace if available, otherwise infers from flows."""
    node_map: dict[str, NavNode] = {p.name: NavNode(name=p.name, page=p) for p in pages}
    has_parent: set[str] = set()

    if trace:
        # Identify root pages (parent is None/missing) upfront so re-visits
        # as children of other pages don't prevent them from being roots.
        root_pages: set[str] = set()
        for entry in trace:
            if not entry.get("parent"):
                root_pages.add(entry.get("page", ""))

        # Accurate: use recorded BFS parent-child relationships
        for entry in trace:
            page_name = entry.get("page", "")
            parent_name = entry.get("parent")
            via_tap = entry.get("via_tap")
            if page_name not in node_map or not parent_name or parent_name not in node_map:
                continue
            # Never assign a parent to a root page — it stays at the top of the tree.
            if page_name in root_pages:
                continue
            child = node_map[page_name]
            parent = node_map[parent_name]
            # Only assign one parent per node to keep a proper tree (no cycles).
            if page_name not in has_parent and child not in parent.children:
                child.via_tap = via_tap
                parent.children.append(child)
                has_parent.add(page_name)
        # When BFS trace is available, only show BFS-explored pages in the tree.
        # Flow targets are AI-generated names that may not match BFS page names,
        # can include back-navigation (parent pages), and cause duplicate/misplaced nodes.
    else:
        # Fallback: infer from page_flows.json with fuzzy name matching
        known: dict[str, ReconPageInfo] = {}
        for p in pages:
            known[p.name] = p
            known[p.title] = p
            clean = re.sub(r'\s*\(\d+\)\s*$', '', p.title).strip()
            if clean:
                known[clean] = p

        def _match(target: str) -> ReconPageInfo | None:
            if target in known:
                return known[target]
            for key, page in known.items():
                if target in key or key in target:
                    return page
            return None

        for page in pages:
            parent = node_map[page.name]
            seen: set[str] = set()
            for flow in page.flows:
                target = flow.target_page
                if target in seen:
                    continue
                seen.add(target)
                matched = _match(target)
                if matched and matched.name != page.name:
                    child = node_map[matched.name]
                    if child not in parent.children:
                        parent.children.append(child)
                        has_parent.add(matched.name)
                else:
                    parent.children.append(NavNode(name=target, page=None))

    return [node_map[p.name] for p in pages if p.name not in has_parent]


def _render_tree_html(nodes: list[NavNode], _visited: frozenset[str] = frozenset()) -> str:
    if not nodes:
        return ""
    items = ""
    for node in nodes:
        if node.page is not None:
            if node.page.name in _visited:
                continue  # cycle guard
            type_label = PAGE_TYPE_LABELS.get(node.page.page_type, node.page.page_type)
            slug = _slug(node.page.name)
            badge = f'<span class="tree-chip">{type_label}</span>' if type_label else ''
            child_visited = _visited | {node.page.name}
            sub = f'<ul>{_render_tree_html(node.children, child_visited)}</ul>' if node.children else ''
            link_cls = "tree-link tree-link-error" if node.page.error else "tree-link"
            error_dot = '<span class="tree-error-dot">⚠</span>' if node.page.error else ''
            items += (
                f'<li class="tree-node">'
                f'<a class="{link_cls}" href="#{slug}">{error_dot}{node.page.title}{badge}</a>'
                f'{sub}</li>'
            )
        else:
            items += (
                f'<li class="tree-node">'
                f'<span class="tree-leaf">· {node.name}</span>'
                f'</li>'
            )
    return items


def _dfs_order(nodes: list[NavNode], depth: int = 0) -> list[tuple[NavNode, int]]:
    """DFS traversal returning (node, depth) pairs in pre-order."""
    result = []
    for node in nodes:
        result.append((node, depth))
        if node.children:
            result.extend(_dfs_order(node.children, depth + 1))
    return result


# ── Recon builder ─────────────────────────────────────────────

def _normalize_error(raw: str | dict | None) -> dict | None:
    """Normalize error to {message, failed_tap?, failed_element?, back_attempts?} or None."""
    if not raw:
        return None
    if isinstance(raw, str):
        return {"message": raw}
    return raw  # already a dict from ProbeAbortedError


def _render_error_html(error: dict | None, annotated_url: str = "") -> str:
    if not error:
        return ""
    msg = error.get("message", "探测中断")
    failed_tap = error.get("failed_tap")
    failed_el = error.get("failed_element", "")
    attempts: list[dict] = error.get("back_attempts", [])

    meta = ""
    if failed_tap and failed_tap > 0 and failed_el:
        meta = f'<div class="error-meta">第 {failed_tap} 个元素「{failed_el}」点击后无法返回</div>'

    timeline = ""
    if attempts:
        steps = ""
        for i, a in enumerate(attempts, 1):
            strategy = a.get("strategy", "?")
            coords = a.get("coords", [])
            coord_str = f"({coords[0]},{coords[1]})" if coords else ""
            result_text = a.get("result", "")
            score = a.get("score")
            score_str = f"{score:.3f}" if score is not None else ""
            score_html = f'<span class="back-score">{score_str}</span>' if score_str else ""
            success = a.get("success", False)
            if strategy == "retry":
                steps += (
                    f'<div class="back-step back-step-retry">'
                    f'<span class="back-num">↻</span>'
                    f'<span class="back-strategy">retry</span>'
                    f'<span class="back-coords"></span>'
                    f'<span class="back-result">{result_text}</span>'
                    f'{score_html}'
                    f'</div>'
                )
                continue
            step_cls = "back-step back-step-ok" if success else "back-step"
            steps += (
                f'<div class="{step_cls}">'
                f'<span class="back-num">{i}</span>'
                f'<span class="back-strategy">{strategy}</span>'
                f'<span class="back-coords">{coord_str}</span>'
                f'<span class="back-result">{result_text}</span>'
                f'{score_html}'
                f'</div>'
            )
        timeline = f'<div class="back-timeline">{steps}</div>'

    img_html = ""
    if annotated_url:
        img_html = (
            f'<div class="error-screenshot">'
            f'<img src="{annotated_url}" title="点击放大" onclick="openModal(this.src)">'
            f'<div class="error-screenshot-caption">返回尝试位置（数字=尝试顺序）</div>'
            f'</div>'
        )

    detail_html = (
        f'{meta}'
        f'<div class="error-body">'
        f'{img_html}'
        f'{timeline}'
        f'</div>'
    ) if (meta or img_html or timeline) else ""

    if detail_html:
        return (
            f'<details class="error-banner">'
            f'<summary class="error-title">⚠ 探测中断：{msg}</summary>'
            f'{detail_html}'
            f'</details>'
        )
    return f'<div class="error-banner"><div class="error-title">⚠ 探测中断：{msg}</div></div>'


class ReconReportBuilder:
    def build(self, log_dir: Path) -> AppReconData:
        # log_dir is either the app dir (logs/recon/微信) or a single page dir
        if (log_dir / "recon_result.json").exists():
            # Single page directory
            page_dirs = [log_dir]
        else:
            # App directory — iterate its page subdirectories
            page_dirs = sorted(p for p in log_dir.iterdir() if p.is_dir())

        app_name = log_dir.name
        pages: list[ReconPageInfo] = []
        total_taps = 0
        total_navigated = 0

        # Load trace.json early (needed for error lookup during page iteration)
        trace_data: list[dict] | None = None
        trace_path = log_dir / "trace.json"
        if trace_path.exists():
            _raw = json.loads(trace_path.read_text(encoding="utf-8"))
            # Support both old format (list) and new format (dict with pages/transitions)
            trace_data = _raw if isinstance(_raw, list) else _raw.get("pages", [])

        # Index trace errors by page name for quick lookup
        trace_errors: dict[str, str] = {}
        if trace_data:
            for entry in trace_data:
                if entry.get("error"):
                    trace_errors[entry["page"]] = entry["error"]

        for pd in page_dirs:
            initial_path = pd / "initial.png"
            if not initial_path.exists():
                continue  # nothing to show at all

            result_path = pd / "recon_result.json"
            result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
            if result:
                app_name = result.get("app_name", app_name)

            # Page identity — prefer recon_result, fall back to initial_result
            page_type = result.get("page_type", "")
            signature = result.get("signature", "")
            elements_count = result.get("elements_count", 0)
            page_title = result.get("page_title", pd.name)
            description = result.get("description", "")

            init_result_path = pd / "initial_result.json"
            if init_result_path.exists():
                init_result = json.loads(init_result_path.read_text(encoding="utf-8"))
                identity = init_result.get("page", {}).get("identity", {})
                if not page_type:
                    page_type = identity.get("page_type", "")
                if not signature:
                    signature = identity.get("signature", "")
                if not page_title or page_title == pd.name:
                    page_title = identity.get("page_title", pd.name)
                if not description:
                    description = init_result.get("page", {}).get("description", "")

            # Annotate screenshot with tap points, save to disk (not embedded in HTML).
            initial_img = _load_img(initial_path)
            raw_taps = result.get("taps", [])
            tap_points = [(t["x"], t["y"], t["index"], t.get("navigated", False)) for t in raw_taps]
            annotated_img = annotate_recon_taps(initial_img, tap_points)
            ann_path = pd / "initial_tap_ann.jpg"
            _save_report_img(annotated_img, ann_path)
            annotated_url = str(ann_path.relative_to(log_dir))

            # Build ReconTap list
            taps: list[ReconTap] = []
            for tap in raw_taps:
                total_taps += 1
                navigated = tap.get("navigated", False)
                if navigated:
                    total_navigated += 1
                tap_path = Path(tap.get("screenshot", ""))
                # Raw tap screenshot already on disk — just use a relative path.
                after_url = str(tap_path.relative_to(log_dir)) if tap_path.is_file() else None

                # Build full back-navigation sequence for navigated taps
                back_seq: list[dict] = []
                if navigated and after_url:
                    tap_idx = tap["index"]
                    tap_dir = pd / "tap"
                    # Step 1: initial page with single tap marker (annotated, save to disk)
                    single_point = [(tap["x"], tap["y"], tap_idx, True)]
                    before_img = annotate_recon_taps(initial_img.copy(), single_point)
                    seq0_path = tap_dir / f"tap_{tap_idx:02d}_seq0.jpg"
                    _save_report_img(before_img, seq0_path)
                    back_seq.append({"src": str(seq0_path.relative_to(log_dir)), "subtitle": "", "success": None})
                    # Step 2: navigated page, annotate with first back-attempt coords if available
                    back_attempts_raw = tap.get("back_attempts", [])
                    if back_attempts_raw and tap_path.is_file():
                        after_img = _load_img(tap_path)
                        after_ann = annotate_back_attempts_img(after_img, [back_attempts_raw[0]])
                        seq1_path = tap_dir / f"tap_{tap_idx:02d}_seq1.jpg"
                        _save_report_img(after_ann, seq1_path)
                        step2_src = str(seq1_path.relative_to(log_dir))
                        first_strategy = back_attempts_raw[0].get('strategy', '')
                        step2_sub = "重新进入子页面" if first_strategy == "forward" else f"回退策略: {first_strategy}"
                    else:
                        step2_src = after_url
                        step2_sub = "已导航"
                    back_seq.append({"src": step2_src, "subtitle": step2_sub, "success": None})
                    # Steps 3+: each back attempt (with screenshot, or retry markers)
                    # Forward steps without a screenshot are collapsed into one terminal
                    # step at the end (they all land on the initial page anyway).
                    pending_forward: list[dict] = []
                    for attempt in tap.get("back_attempts", []):
                        strategy = attempt.get("strategy", "")
                        result_txt = attempt.get("result", "")
                        score = attempt.get("score")
                        score_str = f" {score:.3f}" if score is not None else ""
                        success = attempt.get("success", False)
                        if strategy == "retry":
                            back_seq.append({
                                "src": back_seq[-1]["src"] if back_seq else "",
                                "subtitle": f"↻ {result_txt}{score_str}",
                                "success": None,
                                "is_retry": True,
                            })
                            continue
                        shot = Path(attempt.get("screenshot", ""))
                        if not shot.is_file():
                            if strategy == "forward" and success:
                                pending_forward.append(attempt)
                            continue
                        if strategy == "forward":
                            subtitle = f"{result_txt}（已恢复）"
                        else:
                            subtitle = f"{result_txt}{score_str}"
                        back_seq.append({
                            "src": str(shot.relative_to(log_dir)),
                            "subtitle": subtitle,
                            "success": success,
                        })

                    # Collapse pending no-screenshot forward steps into one terminal step.
                    # Extract the full path: "L0→L1", "L1→L2" → "L0→L1→L2"
                    if pending_forward:
                        steps = [a.get("result", "") for a in pending_forward]
                        levels = [steps[0].split("→")[0]] + [s.split("→")[-1] for s in steps]
                        path_str = "→".join(levels)
                        back_seq.append({
                            "src": str(initial_path.relative_to(log_dir)),
                            "subtitle": f"{path_str}（已恢复）",
                            "success": True,
                        })

                taps.append(ReconTap(
                    index=tap["index"],
                    label=tap.get("label", ""),
                    x=tap.get("x", 0),
                    y=tap.get("y", 0),
                    navigated=navigated,
                    after_url=after_url,
                    back_seq=back_seq,
                ))

            # Load flows
            flows: list[ReconFlow] = []
            flows_path = pd / "page_flows.json"
            if flows_path.exists():
                flows_data = json.loads(flows_path.read_text(encoding="utf-8"))
                for f in flows_data.get("flows", []):
                    flows.append(ReconFlow(
                        target_page=f.get("target_page", ""),
                        target_description=f.get("target_description", ""),
                        flow_description=f.get("flow_description", ""),
                    ))

            # Load knowledge
            knowledge = ""
            knowledge_path = pd / "knowledge.md"
            if knowledge_path.exists():
                knowledge = knowledge_path.read_text(encoding="utf-8")

            page_error = _normalize_error(trace_errors.get(
                pd.name,
                None if result_path.exists() else "探测中断，结果未保存",
            ))

            # Annotate failed-tap screenshot with back-attempt coords
            error_annotated_url = ""
            if page_error:
                back_attempts = page_error.get("back_attempts", [])
                if back_attempts:
                    failed_tap_idx = page_error.get("failed_tap", -1)
                    shot_bytes: bytes | None = None
                    if failed_tap_idx and failed_tap_idx > 0:
                        for tap in raw_taps:
                            if tap.get("index") == failed_tap_idx:
                                tp = Path(tap.get("screenshot", ""))
                                if tp.is_file():
                                    shot_bytes = tp.read_bytes()
                                break
                    if shot_bytes is None:
                        shot_bytes = initial_path.read_bytes()
                    err_img = Image.open(io.BytesIO(shot_bytes)).convert("RGBA")
                    err_img = annotate_back_attempts_img(err_img, back_attempts)
                    err_ann_path = pd / "error_tap_ann.jpg"
                    _save_report_img(err_img, err_ann_path)
                    error_annotated_url = str(err_ann_path.relative_to(log_dir))

            pages.append(ReconPageInfo(
                name=pd.name,
                title=page_title,
                page_type=page_type,
                description=description,
                elements_count=elements_count,
                signature=signature,
                annotated_url=annotated_url,
                taps=taps,
                flows=flows,
                knowledge=knowledge,
                error=page_error,
                error_annotated_url=error_annotated_url,
            ))

        data = AppReconData(
            app_name=app_name,
            pages=pages,
            stats={
                "pages": len(pages),
                "taps_probed": total_taps,
                "navigated": total_navigated,
                "no_change": total_taps - total_navigated,
            },
            trace=trace_data,
        )
        return data


# ── Runner builder ─────────────────────────────────────────────

class RunnerReportBuilder:
    def build(self, run_dir: Path) -> ReportData:
        data = ReportData(title=run_dir.name)
        ctx_path = run_dir / "context.json"
        if not ctx_path.exists():
            return data

        ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
        data.title = ctx.get("goal", run_dir.name)

        page = ReportPage(title="Execution")
        turns = ctx.get("turns", [])

        total_actions = 0
        total_executed = 0

        for turn in turns:
            idx = turn.get("index", 0)
            ad = turn.get("action_decision") or {}
            action = ad.get("action") or {}
            atype = action.get("action_type", "none")
            x = action.get("x")
            y = action.get("y")
            desc = action.get("description", "")
            sup = turn.get("supervisor") or {}
            summary = sup.get("summary", "")
            executed = turn.get("executed", False)

            total_actions += 1
            if executed:
                total_executed += 1

            ss_path = run_dir / f"screenshot_turn_{idx}.png"
            if ss_path.exists() and x is not None and y is not None:
                img = _load_img(ss_path)
                annotated_img = annotate_tap(img, [(x, y, idx)])
                ann_path = run_dir / f"screenshot_turn_{idx}_ann.jpg"
                _save_report_img(annotated_img, ann_path)
                annotated_url = ann_path.name
            elif ss_path.exists():
                annotated_url = ss_path.name
            else:
                annotated_url = ""

            status = "✓" if executed else "✗"
            if atype == "none":
                status = "— skip"

            page.steps.append(ReportStep(
                label=f"Turn {idx}",
                action_type=atype,
                x=x,
                y=y,
                description=desc or summary,
                annotated_before_url=annotated_url,
                after_url=None,
                status=status,
                timestamp=turn.get("timestamp", ""),
            ))

        data.pages.append(page)
        data.stats = {
            "turns": len(turns),
            "executed": total_executed,
        }
        return data


# ── Recon HTML generator ───────────────────────────────────────

PAGE_TYPE_LABELS: dict[str, str] = {
    "list": "列表",
    "chat": "聊天",
    "detail": "详情",
    "form": "表单",
    "modal": "弹窗",
    "other": "其他",
}

RECON_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{app_name} — Recon Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  :root {{
    --green: #22c55e;
    --gray: #9ca3af;
    --blue: #3b82f6;
    --bg: #f1f5f9;
    --card: #ffffff;
    --border: #e2e8f0;
    --text: #1e293b;
    --muted: #64748b;
    --radius: 12px;
  }}
  body {{ font-family: -apple-system, "PingFang SC", "Helvetica Neue", sans-serif; background: var(--bg); color: var(--text); }}

  /* ── Layout ── */
  .layout {{ display: flex; min-height: 100vh; }}
  .sidebar {{
    width: 220px; flex-shrink: 0; position: sticky; top: 0; height: 100vh;
    background: var(--card); border-right: 1px solid var(--border);
    overflow-y: auto; padding: 20px 0;
  }}
  .main {{ flex: 1; padding: 24px; max-width: 900px; }}

  /* ── Sidebar ── */
  .sidebar-title {{ font-size: 10px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; padding: 0 16px 10px; }}
  .sidebar-stats {{ padding: 14px 16px; margin-top: 8px; border-top: 1px solid var(--border); font-size: 12px; color: var(--muted); line-height: 2; }}
  .sidebar-stats strong {{ color: var(--text); }}

  /* ── Nav tree ── */
  .nav-tree, .nav-tree ul {{ list-style: none; margin: 0; padding: 0; }}
  .nav-tree {{ padding: 0 8px; }}
  .nav-tree > li {{ padding-left: 4px; }}
  .nav-tree ul {{
    margin-left: 14px;
    padding-left: 10px;
    border-left: 1px solid var(--border);
  }}
  .tree-node {{ position: relative; padding: 1px 0; }}
  .nav-tree ul > .tree-node::before {{
    content: "";
    position: absolute;
    left: -11px;
    top: 14px;
    width: 10px;
    height: 1px;
    background: var(--border);
  }}
  .tree-link {{
    display: flex; align-items: center; gap: 5px;
    padding: 5px 8px; border-radius: 6px;
    font-size: 12px; color: var(--text); text-decoration: none;
    transition: background 0.12s;
  }}
  .tree-link:hover {{ background: var(--bg); }}
  .tree-link-error {{ color: #dc2626; }}
  .tree-link-error:hover {{ background: #fef2f2; }}
  .tree-error-dot {{ font-size: 10px; flex-shrink: 0; }}
  .tree-leaf {{
    display: block; padding: 4px 8px;
    font-size: 11px; color: var(--gray); font-style: italic;
  }}
  .tree-chip {{
    font-size: 9px; padding: 1px 4px; border-radius: 3px; flex-shrink: 0;
    background: #e0e7ff; color: #4338ca;
  }}

  /* ── Header ── */
  .header {{ margin-bottom: 24px; }}
  .header h1 {{ font-size: 22px; font-weight: 700; margin-bottom: 4px; }}
  .header .subtitle {{ font-size: 13px; color: var(--muted); }}

  /* ── Tree connectors (right panel) ── */
  .tree-item {{ position: relative; }}
  .tree-children {{
    margin-left: 20px;
    padding-left: 28px;
    border-left: 2px solid var(--border);
  }}
  .tree-children > .tree-item {{ position: relative; }}
  .tree-children > .tree-item::before {{
    content: "";
    position: absolute;
    left: -28px;
    top: 32px;
    width: 28px;
    height: 2px;
    background: var(--border);
  }}

  /* ── Page card ── */
  .page-card {{
    background: var(--card); border-radius: var(--radius);
    border: 1px solid var(--border); margin-bottom: 16px;
    overflow: hidden;
  }}
  .page-card-error {{ border-color: #fca5a5; border-left: 3px solid #ef4444; }}
  .page-card-error .page-card-header {{ background: #fff5f5; }}
  .page-breadcrumb {{
    padding: 6px 20px;
    font-size: 11px; color: var(--muted);
    background: #f8fafc; border-bottom: 1px solid var(--border);
  }}
  .bc-sep {{ color: #cbd5e1; margin: 0 5px; }}
  .bc-current {{ color: var(--text); font-weight: 600; }}
  .page-card-header {{
    padding: 12px 20px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  }}
  .page-card-header h2 {{ font-size: 15px; font-weight: 600; }}
  .type-badge {{
    font-size: 11px; padding: 2px 8px; border-radius: 20px; font-weight: 500;
    background: #dbeafe; color: #1d4ed8;
  }}
  .page-sig {{ font-size: 11px; color: var(--muted); flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .via-tap {{ font-size: 11px; color: #7c3aed; background: #ede9fe; padding: 2px 7px; border-radius: 10px; white-space: nowrap; flex-shrink: 0; }}
  .page-card-desc {{ padding: 10px 20px; font-size: 13px; color: var(--muted); border-bottom: 1px solid var(--border); background: #f8fafc; }}

  /* ── Page body ── */
  .page-card-body {{ display: flex; gap: 0; }}
  .screenshot-col {{
    width: 260px; flex-shrink: 0; padding: 16px;
    border-right: 1px solid var(--border);
  }}
  .screenshot-col img {{
    width: 100%; border-radius: 8px; border: 1px solid var(--border);
    cursor: zoom-in; display: block;
  }}
  .legend {{ display: flex; gap: 12px; margin-top: 8px; font-size: 11px; color: var(--muted); justify-content: center; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; vertical-align: middle; margin-right: 3px; }}
  .info-col {{ flex: 1; min-width: 0; padding: 16px; display: flex; flex-direction: column; gap: 16px; }}

  /* ── Section labels ── */
  .section-label {{ font-size: 11px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }}

  /* ── Tap table ── */
  .tap-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .tap-table tr {{ border-bottom: 1px solid var(--border); }}
  .tap-table tr:last-child {{ border-bottom: none; }}
  .tap-table td {{ padding: 6px 4px; vertical-align: middle; }}
  .tap-num {{ width: 24px; text-align: center; font-size: 11px; font-weight: 600; color: var(--muted); }}
  .tap-label {{ font-weight: 500; max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .tap-status {{ font-size: 12px; }}
  .tap-nav {{ color: var(--green); }}
  .tap-none {{ color: var(--gray); }}
  .tap-target {{
    font-size: 11px; color: #15803d; background: #f0fdf4;
    padding: 1px 6px; border-radius: 8px; margin-left: 4px;
    border: 1px solid #bbf7d0; white-space: nowrap;
  }}
  .flow-extra-row td {{ color: var(--muted); font-style: italic; }}
  .tap-preview-btn {{
    font-size: 11px; padding: 2px 8px; border-radius: 4px; border: 1px solid var(--border);
    background: var(--bg); color: var(--muted); cursor: pointer;
    transition: all 0.15s; white-space: nowrap;
  }}
  .tap-preview-btn:hover {{ background: #dbeafe; border-color: #93c5fd; color: #1d4ed8; }}

  /* ── Error banner ── */
  .error-banner {{
    font-size: 12px; color: #991b1b;
    background: #fef2f2; border-bottom: 1px solid #fecaca;
  }}
  .error-title {{
    font-weight: 600; list-style: none;
    padding: 10px 20px;
    display: flex; align-items: center; gap: 6px; cursor: pointer;
  }}
  .error-title::-webkit-details-marker {{ display: none; }}
  .error-title::before {{ content: "▶"; font-size: 9px; transition: transform 0.15s; margin-right: 2px; }}
  details.error-banner[open] .error-title::before {{ transform: rotate(90deg); }}
  .error-meta {{ color: #b91c1c; margin-bottom: 8px; font-size: 11px; padding: 0 20px; }}
  .error-body {{ padding: 0 20px 12px; }}
  .error-body {{ display: flex; gap: 16px; align-items: flex-start; }}
  .error-screenshot {{ flex-shrink: 0; width: 120px; }}
  .error-screenshot img {{
    width: 100%; border-radius: 6px; border: 1px solid #fecaca;
    cursor: zoom-in; display: block;
  }}
  .error-screenshot-caption {{ font-size: 10px; color: #b91c1c; text-align: center; margin-top: 4px; }}
  .back-timeline {{ display: flex; flex-direction: column; gap: 4px; flex: 1; }}
  .back-step {{
    display: flex; align-items: center; gap: 8px;
    font-size: 11px; padding: 4px 8px; border-radius: 6px;
    background: #fff7f7; border: 1px solid #fecaca;
  }}
  .back-step-ok {{ background: #f0fdf4; border-color: #bbf7d0; color: #166534; }}
  .back-step-retry {{ background: #fff7ed; border-color: #fed7aa; color: #9a3412; }}
  .back-num {{ width: 18px; height: 18px; border-radius: 50%; background: #fecaca; color: #991b1b; font-weight: 700; font-size: 10px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }}
  .back-step-ok .back-num {{ background: #bbf7d0; color: #166534; }}
  .back-step-retry .back-num {{ background: #fed7aa; color: #c2410c; font-size: 12px; }}
  .back-strategy {{ font-weight: 600; width: 70px; flex-shrink: 0; color: #991b1b; }}
  .back-step-ok .back-strategy {{ color: #166534; }}
  .back-step-retry .back-strategy {{ color: #ea580c; }}
  .back-coords {{ color: #64748b; width: 72px; flex-shrink: 0; font-family: monospace; }}
  .back-result {{ flex: 1; color: #64748b; }}
  .back-score {{ font-family: monospace; color: #b91c1c; margin-left: auto; flex-shrink: 0; }}
  .back-step-ok .back-score {{ color: #166534; }}

  /* ── Flows ── */
  .flows-list {{ display: flex; flex-direction: column; gap: 8px; }}
  .flow-item {{ padding: 8px 10px; background: #f0fdf4; border-radius: 8px; border: 1px solid #bbf7d0; }}
  .flow-target {{ font-size: 13px; font-weight: 600; color: #15803d; }}
  .flow-desc {{ font-size: 12px; color: #166534; margin-top: 2px; line-height: 1.4; }}

  /* ── Knowledge ── */
  details {{ border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
  summary {{
    padding: 8px 12px; font-size: 12px; font-weight: 600; color: var(--muted);
    cursor: pointer; user-select: none; list-style: none; display: flex; align-items: center; gap: 6px;
    background: #f8fafc;
  }}
  summary::before {{ content: "▶"; font-size: 9px; transition: transform 0.15s; }}
  details[open] summary::before {{ transform: rotate(90deg); }}
  .knowledge-body {{ padding: 12px; font-size: 12px; color: #475569; line-height: 1.7; white-space: pre-wrap; background: white; }}

  /* ── Modal ── */
  .modal {{
    display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.88); z-index: 1000;
    justify-content: center; align-items: center; gap: 16px;
  }}
  .modal.show {{ display: flex; }}
  #modal-simple {{ display: flex; align-items: center; gap: 16px; }}
  .modal-panel {{
    background: white; border-radius: 12px; padding: 12px;
    display: flex; flex-direction: column; align-items: center; gap: 8px;
    max-height: 90vh; border: 2px solid transparent;
  }}
  .modal-panel img {{ max-height: calc(90vh - 60px); max-width: 320px; border-radius: 8px; object-fit: contain; }}
  .modal-label {{ font-size: 11px; color: #64748b; text-align: center; max-width: 280px; }}
  .modal-close {{
    position: fixed; top: 20px; right: 24px; font-size: 28px; color: white;
    cursor: pointer; line-height: 1; opacity: 0.8; z-index: 1001;
  }}
  .modal-close:hover {{ opacity: 1; }}
  .tap-seq-btn {{ background: #ede9fe; border-color: #c4b5fd; color: #6d28d9; }}
  .tap-seq-btn:hover {{ background: #ddd6fe; border-color: #a78bfa; color: #5b21b6; }}
</style>
</head>
<body>

<div class="layout">
  <nav class="sidebar">
    <div class="sidebar-title">{app_name}</div>
    {sidebar_items}
    <div class="sidebar-stats">
      <strong>{pages}</strong> 页面<br>
      <strong>{taps_probed}</strong> 点击测试<br>
      <span style="color:var(--green)">●</span> <strong>{navigated}</strong> 导航成功<br>
      <span style="color:var(--gray)">●</span> <strong>{no_change}</strong> 无变化
    </div>
  </nav>

  <main class="main">
    <div class="header">
      <h1>{app_name}</h1>
      <div class="subtitle">Recon Report · {pages} 个页面 · {taps_probed} 次点击探测</div>
    </div>

    {pages_html}
  </main>
</div>

<!-- Modal -->
<div class="modal" id="modal" onclick="closeModal(event)">
  <div class="modal-close" onclick="document.getElementById('modal').classList.remove('show')">✕</div>
  <!-- Simple zoom mode -->
  <div id="modal-simple" style="display:none">
    <div class="modal-panel" id="modal-before">
      <img id="modal-before-img" src="">
      <div class="modal-label">操作前</div>
    </div>
    <div id="modal-arrow" style="font-size:28px;color:white;display:none">→</div>
    <div class="modal-panel" id="modal-after-panel" style="display:none">
      <img id="modal-after-img" src="">
      <div class="modal-label" id="modal-after-label"></div>
    </div>
  </div>
  <!-- Sequence mode -->
  <div id="modal-seq-wrap" style="display:none;overflow-x:auto;max-width:96vw;padding:8px">
    <div id="modal-seq" style="display:flex;align-items:flex-start;gap:12px;min-width:max-content"></div>
  </div>
</div>

<script>
function openModal(beforeSrc, afterSrc, label) {{
  document.getElementById('modal-simple').style.display = 'flex';
  document.getElementById('modal-seq-wrap').style.display = 'none';
  document.getElementById('modal-before-img').src = beforeSrc;
  var afterPanel = document.getElementById('modal-after-panel');
  var arrow = document.getElementById('modal-arrow');
  if (afterSrc) {{
    document.getElementById('modal-after-img').src = afterSrc;
    document.getElementById('modal-after-label').textContent = label || '操作后';
    afterPanel.style.display = '';
    arrow.style.display = '';
  }} else {{
    afterPanel.style.display = 'none';
    arrow.style.display = 'none';
  }}
  document.getElementById('modal').classList.add('show');
}}
function openSeq(key) {{
  var steps = (window._seqs || {{}})[key] || [];
  var seq = document.getElementById('modal-seq');
  seq.innerHTML = '';
  steps.forEach(function(step, i) {{
    if (i > 0) {{
      var arr = document.createElement('div');
      arr.style.cssText = 'font-size:22px;color:rgba(255,255,255,0.5);align-self:center;flex-shrink:0';
      arr.textContent = '→';
      seq.appendChild(arr);
    }}
    var panel = document.createElement('div');
    panel.className = 'modal-panel';
    if (step.is_retry) panel.style.borderColor = 'rgba(251,146,60,0.6)';
    else if (step.success === true) panel.style.borderColor = 'rgba(34,197,94,0.5)';
    else if (step.success === false) panel.style.borderColor = 'rgba(239,68,68,0.4)';
    var img = document.createElement('img');
    img.src = step.src;
    img.style.cursor = 'default';
    if (step.is_retry) img.style.opacity = '0.35';
    var lbl = document.createElement('div');
    lbl.className = 'modal-label';
    lbl.style.whiteSpace = 'nowrap';
    lbl.textContent = '步骤 ' + (i + 1) + (step.subtitle ? ': ' + step.subtitle : '');
    if (step.is_retry) lbl.style.color = '#fb923c';
    else if (step.success === true) lbl.style.color = '#4ade80';
    else if (step.success === false) lbl.style.color = '#f87171';
    panel.appendChild(img);
    panel.appendChild(lbl);
    seq.appendChild(panel);
  }});
  document.getElementById('modal-simple').style.display = 'none';
  document.getElementById('modal-seq-wrap').style.display = '';
  document.getElementById('modal').classList.add('show');
}}
function showTapResult(btn) {{
  var pageCard = btn.closest('.page-card');
  var annotatedImg = pageCard.querySelector('.screenshot-col img');
  openModal(annotatedImg.src, btn.dataset.after || null, btn.dataset.label);
}}
function closeModal(e) {{
  if (e.target === document.getElementById('modal')) {{
    document.getElementById('modal').classList.remove('show');
  }}
}}
document.addEventListener('keydown', function(e) {{
  if (e.key === 'Escape') document.getElementById('modal').classList.remove('show');
}});
</script>
</body>
</html>
"""


def _slug(name: str) -> str:
    return re.sub(r"[^\w]", "-", name)


def _render_page_card_html(node: NavNode, path: list[str]) -> str:
    """Recursively render a page card and all its explored children."""
    if node.page is None:
        return ""

    page = node.page
    type_label = PAGE_TYPE_LABELS.get(page.page_type, page.page_type)
    slug = _slug(page.name)
    path_with_self = path + [page.title]

    # Breadcrumb: full path for any non-root page
    breadcrumb_html = ""
    if path:
        parts = ""
        for seg in path:
            parts += f'<span class="bc-seg">{seg}</span><span class="bc-sep">›</span>'
        parts += f'<span class="bc-current">{page.title}</span>'
        breadcrumb_html = f'<div class="page-breadcrumb">{parts}</div>'

    via_html = ""
    if node.via_tap:
        via_html = f'<span class="via-tap">← {node.via_tap}</span>'

    # Match flows to navigated taps by checking if tap.label appears in flow_description
    flow_by_tap: dict[int, ReconFlow] = {}
    used_flows: set[int] = set()
    for tap in page.taps:
        if not tap.navigated or not tap.label:
            continue
        for fi, flow in enumerate(page.flows):
            if fi not in used_flows and tap.label in flow.flow_description:
                flow_by_tap[tap.index] = flow
                used_flows.add(fi)
                break
    # Merged tap + flow table
    tap_rows = ""
    seq_scripts = ""
    for tap in page.taps:
        if tap.navigated:
            matched = flow_by_tap.get(tap.index)
            target_chip = f'<span class="tap-target">→ {matched.target_page}</span>' if matched else ''
            status_html = f'<span class="tap-nav">✓ 导航成功</span>{target_chip}'
            seq_key = f"{_slug(page.name)}-{tap.index}"
            if len(tap.back_seq) > 1:
                # Full sequence available — emit JS and use openSeq button
                import json as _json
                seq_json = _json.dumps(tap.back_seq)
                seq_scripts += f'_seqs[{_json.dumps(seq_key)}]={seq_json};\n'
                btn = (
                    f'<button class="tap-preview-btn tap-seq-btn"'
                    f" onclick=\"openSeq('{seq_key}')\">查看全过程 ({len(tap.back_seq)})</button>"
                )
            elif tap.after_url:
                tap_lbl = f"tap_{tap.index}: {tap.label}"
                btn = (
                    f'<button class="tap-preview-btn"'
                    f' data-label="{tap_lbl}" data-after="{tap.after_url}"'
                    f' onclick="showTapResult(this)">查看结果</button>'
                )
            else:
                btn = ""
        else:
            status_html = '<span class="tap-none">— 无变化</span>'
            btn = ""
        tap_rows += f"""
            <tr>
              <td class="tap-num">{tap.index}</td>
              <td class="tap-label" title="{tap.label}">{tap.label or "—"}</td>
              <td class="tap-status">{status_html}</td>
              <td>{btn}</td>
            </tr>"""

    nav_count = sum(1 for t in page.taps if t.navigated)
    matched_flows_count = len(used_flows)
    flows_label = f" · {matched_flows_count} 条路径" if matched_flows_count else ""
    tap_section = f"""
        <div>
          <div class="section-label">点击探测 ({nav_count}/{len(page.taps)} 导航成功{flows_label})</div>
          <table class="tap-table">{tap_rows}</table>
        </div>"""

    flows_section = ""

    knowledge_section = ""
    if page.knowledge.strip():
        safe_knowledge = (
            page.knowledge
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        knowledge_section = f"""
            <details>
              <summary>页面知识摘要</summary>
              <div class="knowledge-body">{safe_knowledge}</div>
            </details>"""

    card_cls = "page-card page-card-error" if page.error else "page-card"
    seq_init = f"<script>if(!window._seqs)window._seqs={{}};\n{seq_scripts}</script>" if seq_scripts else ""
    card_html = f"""
        {seq_init}<div class="{card_cls}" id="{slug}">
          {breadcrumb_html}
          <div class="page-card-header">
            <h2>{page.title}</h2>
            <span class="type-badge">{type_label}</span>
            {via_html}
            <span class="page-sig">{page.signature}</span>
          </div>
          {'<div class="page-card-desc">' + page.description + '</div>' if page.description else ''}
          {_render_error_html(page.error, page.error_annotated_url)}
          <div class="page-card-body">
            <div class="screenshot-col">
              <img src="{page.annotated_url}" onclick="openModal(this.src)" title="点击放大">
              <div class="legend">
                <span><span class="legend-dot" style="background:var(--green)"></span>导航成功</span>
                <span><span class="legend-dot" style="background:var(--gray)"></span>无变化</span>
              </div>
            </div>
            <div class="info-col">
              {tap_section}
              {flows_section}
              {knowledge_section}
            </div>
          </div>
        </div>"""

    # Recursively render explored children
    children_html = "".join(
        _render_page_card_html(child, path_with_self)
        for child in node.children
        if child.page is not None
    )
    children_section = f'<div class="tree-children">{children_html}</div>' if children_html else ""

    return f'<div class="tree-item">{card_html}{children_section}</div>'


def generate_recon_html(data: AppReconData) -> str:
    roots = _build_nav_tree(data.pages, trace=data.trace)
    tree_html = _render_tree_html(roots)
    sidebar_items = f'<ul class="nav-tree">{tree_html}</ul>'

    pages_html = "".join(_render_page_card_html(root, []) for root in roots)

    return RECON_HTML_TEMPLATE.format(
        app_name=data.app_name,
        sidebar_items=sidebar_items,
        pages_html=pages_html,
        **data.stats,
    )


# ── Runner HTML generator ──────────────────────────────────────

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, "PingFang SC", sans-serif; background: #f5f5f5; padding: 20px; }}
  .header {{ max-width: 900px; margin: 0 auto 24px; padding: 20px; background: #fff; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .header h1 {{ font-size: 20px; margin-bottom: 8px; }}
  .stats {{ color: #666; font-size: 14px; }}
  .page {{ max-width: 900px; margin: 0 auto 24px; padding: 20px; background: #fff; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .page h2 {{ font-size: 16px; color: #333; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid #eee; }}
  .step {{ margin-bottom: 20px; padding: 12px; border: 1px solid #eee; border-radius: 8px; }}
  .step-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }}
  .step-idx {{ display: inline-block; width: 28px; height: 28px; line-height: 28px; text-align: center; border-radius: 50%; font-size: 13px; font-weight: bold; color: #fff; }}
  .step-label {{ font-size: 14px; font-weight: 500; }}
  .step-desc {{ font-size: 13px; color: #666; margin-bottom: 8px; }}
  .step-status {{ font-size: 12px; font-weight: 500; margin-left: auto; }}
  .step-images {{ display: flex; gap: 8px; }}
  .step-images img {{ max-height: 300px; border-radius: 6px; cursor: pointer; border: 1px solid #eee; }}
  .img-label {{ font-size: 11px; color: #999; text-align: center; margin-top: 2px; }}
  .modal {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 999; justify-content: center; align-items: center; }}
  .modal.show {{ display: flex; }}
  .modal img {{ max-width: 90%; max-height: 90%; }}
  .arrow {{ display: flex; align-items: center; justify-content: center; font-size: 24px; color: #ccc; padding: 0 4px; }}
  .grid {{ display: flex; flex-wrap: nowrap; overflow-x: auto; gap: 12px; padding-bottom: 8px; }}
  .grid .card {{ min-width: 240px; max-width: 240px; flex-shrink: 0; padding: 8px; }}
  .grid .card-header {{ margin-bottom: 4px; }}
  .card-seq {{ font-size: 12px; color: #aaa; margin-right: 2px; }}
  .card {{ border: 1px solid #eee; border-radius: 8px; padding: 12px; }}
  .card-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }}
  .card-images {{ display: flex; justify-content: center; }}
  .card-images img {{ width: 100%; border-radius: 6px; cursor: pointer; border: 1px solid #eee; }}
  .step-time {{ font-size: 11px; color: #aaa; margin-left: auto; margin-right: 4px; }}
</style>
</head>
<body>

<div class="header">
  <h1>{title}</h1>
  <div class="stats">{stats}</div>
</div>

{pages_html}

<div class="modal" id="modal" onclick="this.classList.remove('show')">
  <img id="modal-img" src="">
</div>
<script>
document.querySelectorAll('.step-images img').forEach(img => {{
  img.onclick = () => {{
    document.getElementById('modal-img').src = img.src;
    document.getElementById('modal').classList.add('show');
  }};
}});
</script>
</body>
</html>
"""


def color_for_type(action_type: str) -> tuple[int, int, int]:
    return ACTION_COLORS.get(action_type, DEFAULT_COLOR)


def color_hex(color: tuple[int, int, int]) -> str:
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"


def generate_html(data: ReportData, grid: bool = False) -> str:
    stats_parts = [f"{k}: {v}" for k, v in data.stats.items()]
    stats_str = " | ".join(stats_parts)

    pages_html = ""
    for page in data.pages:
        steps_html = ""
        for step_idx, step in enumerate(page.steps, 1):
            color = color_hex(color_for_type(step.action_type))
            images_html = ""

            if step.annotated_before_url:
                images_html += f'<div><img src="{step.annotated_before_url}" alt="before"></div>'
                if step.after_url:
                    images_html += '<div class="arrow">→</div>'
                    images_html += f'<div><img src="{step.after_url}" alt="after"><div class="img-label">结果</div></div>'

            if grid:
                time_str = ""
                if step.timestamp:
                    time_str = step.timestamp.split("T")[1][:8]
                seq = f'{step_idx}. '
                steps_html += f"""
                <div class="card">
                  <div class="card-header">
                    <span class="card-seq">{seq}</span>
                    <span class="step-desc">{step.description}</span>
                    <span class="step-time">{time_str}</span>
                    <span class="step-status">{step.status}</span>
                  </div>
                  <div class="card-images">{images_html}</div>
                </div>"""
            else:
                steps_html += f"""
                <div class="step">
                  <div class="step-header">
                    <span class="step-idx" style="background:{color}">{step.label}</span>
                    <span class="step-desc">{step.description}</span>
                    <span class="step-status">{step.status}</span>
                  </div>
                  <div class="step-images">{images_html}</div>
                </div>"""

        container_cls = "grid" if grid else ""
        pages_html += f"""
        <div class="page">
          <h2>{page.title}</h2>
          <div class="{container_cls}">{steps_html}</div>
        </div>"""

    return HTML_TEMPLATE.format(
        title=data.title,
        stats=stats_str,
        pages_html=pages_html,
    )


def save_report(data: ReportData, output_path: Path, grid: bool = False) -> Path:
    html = generate_html(data, grid=grid)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def save_recon_report(data: AppReconData, output_path: Path) -> Path:
    html = generate_recon_html(data)
    output_path.write_text(html, encoding="utf-8")
    return output_path
