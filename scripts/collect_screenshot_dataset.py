"""Screenshot dataset collector for fingerprint/similarity analysis.

Pure DFS walk: every page gets screenshot saved, no dedup, no knowledge.
Captures every page transition with before/after screenshots + similarity scores.

Output structure:

    logs/dataset/{app}/
    ├── pages/
    │   ├── 000_聊天列表/
    │   │   ├── screenshot.png
    │   │   ├── elements.json
    │   │   └── taps/
    │   │       ├── tap_01_通讯录_after.png
    │   │       └── tap_02_微信_after.png
    │   └── ...
    └── dataset.json   {pages: [{idx, name, pairwise, taps: [{label, navigated, similarity, after_path, child_idx}]}]}

Usage:
    uv run python scripts/collect_screenshot_dataset.py 微信 --depth 2
    uv run python scripts/collect_screenshot_dataset.py 微信 --depth 2 --sample 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOG_ROOT = ROOT / "logs" / "dataset"


class _TeeLogger:
    """Tee stdout and stderr to a log file while keeping terminal output."""

    def __init__(self, log_path: Path):
        self._log = log_path.open("a", encoding="utf-8")
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self  # type: ignore[assignment]
        sys.stderr = self  # type: ignore[assignment]

    def write(self, data: str) -> int:
        self._stdout.write(data)
        self._log.write(data)
        return len(data)

    def flush(self) -> None:
        self._stdout.flush()
        self._log.flush()

    def __getattr__(self, name: str):
        return getattr(self._stdout, name)


@dataclass
class TapRecord:
    index: int
    label: str
    x: float
    y: float
    navigated: bool
    similarity: float           # raw similarity between before and after
    after_path: str = ""       # relative path to after-tap screenshot
    child_page_idx: int = -1   # index into pages if navigated to a new page


@dataclass
class PageEntry:
    idx: int
    name: str
    dir_name: str
    pairwise: list[dict] = field(default_factory=list)
    taps: list[dict] = field(default_factory=list)


def collect(app: str, depth: int = 0, sample: int = 0) -> None:
    from policy_expr.perception import LivePhoneSession
    from policy_expr.recon.page_parser import PageParser
    from policy_expr.recon.page_compare import PageComparator
    from policy_expr.executor import logical_xy

    out_dir = LOG_ROOT / app
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    # Tee stdout/stderr to log file
    log_path = out_dir / "log.txt"
    _TeeLogger(log_path)

    # Preload GUIClip model
    print("预加载 GUIClip 模型...")
    from policy_expr.recon.back_nav import _get_identity_comp
    import io
    from PIL import Image
    dummy = io.BytesIO()
    Image.new('RGB', (1, 1), color='white').save(dummy, format='PNG')
    id_comp = _get_identity_comp()
    id_comp.raw_similarity(dummy.getvalue(), dummy.getvalue())
    print("GUIClip 模型就绪")

    comparator = PageComparator()
    parser = PageParser()

    entries: list[PageEntry] = []
    all_screenshots: list[tuple[int, str, bytes]] = []  # (idx, name, png_bytes)
    visited: set[str] = set()

    def _page_seq(idx: int, name: str) -> str:
        return f"{idx:03d}_{name}"

    def _save_page(page_name: str, png_bytes: bytes) -> int:
        """Save screenshot for a page. Returns index."""
        idx = len(entries)

        dir_name = _page_seq(idx, page_name)
        page_dir = pages_dir / dir_name
        page_dir.mkdir(parents=True, exist_ok=True)

        (page_dir / "screenshot.png").write_bytes(png_bytes)

        pairwise = []
        for prev_idx, prev_name, prev_bytes in all_screenshots:
            sim = comparator.raw_similarity(prev_bytes, png_bytes)
            pairwise.append({"idx": prev_idx, "name": prev_name, "similarity": round(sim, 4)})

        entry = PageEntry(idx=idx, name=page_name, dir_name=dir_name, pairwise=pairwise)
        entries.append(entry)
        all_screenshots.append((idx, page_name, png_bytes))

        _save_dataset(out_dir, entries)
        return idx

    def _dfs(phone, nav_stack, remaining_depth: int, path: list[str]) -> None:
        """DFS step: screenshot → parse once → save → tap each → recurse."""
        png_bytes = phone.screenshot()

        knowledge = None
        page_name = f"page_{len(entries)}"
        try:
            knowledge = parser.analyze_screen(png_bytes)
            desc = knowledge.page.description[:20].strip()
            page_name = re.sub(r'[\\/:*?"<>|\s]+', '_', desc) or page_name
        except Exception:
            pass

        visit_key = "/".join(path + [page_name])
        is_revisit = visit_key in visited
        visited.add(visit_key)

        idx = _save_page(page_name, png_bytes)
        tag = " ( revisit)" if is_revisit else ""
        print(f"  [{idx}] {page_name} (depth={len(path)}){tag}")

        page_dir = pages_dir / _page_seq(idx, page_name)

        if knowledge:
            elements = [
                {"label": a.label, "element_type": a.element_type,
                 "x": a.center_xy[0], "y": a.center_xy[1]}
                for a in knowledge.areas
            ]
            (page_dir / "elements.json").write_text(
                json.dumps(elements, ensure_ascii=False, indent=2)
            )

        if remaining_depth <= 0 or not knowledge:
            if remaining_depth <= 0:
                print(f"    depth=0, 不继续探测子元素")
            return

        areas = [a for a in knowledge.areas if a.element_type != "back_button"]
        if sample > 0 and sample < len(areas):
            import random
            areas = random.sample(areas, sample)

        print(f"    {len(areas)} 个可交互元素")

        tap_dir = page_dir / "taps"
        tap_dir.mkdir(parents=True, exist_ok=True)

        for i, area in enumerate(areas, 1):
            ax, ay = area.center_xy
            lx, ly = logical_xy(ax, ay)
            print(f"    [{i}/{len(areas)}] 「{area.label}」 @ ({lx:.0f},{ly:.0f})")

            phone.client.tap(lx, ly)
            time.sleep(2.0)

            after_bytes = phone.screenshot()
            if not after_bytes:
                from policy_expr.perception import try_resume_mac
                print(f"      Mac 弹窗阻断，关闭后跳过")
                try_resume_mac(phone.client)
                continue
            safe_label = area.label.replace("/", "_").replace(":", "_")[:20]
            after_path = f"{_page_seq(idx, page_name)}/taps/tap_{i:02d}_{safe_label}.png"
            (tap_dir / f"tap_{i:02d}_{safe_label}.png").write_bytes(after_bytes)

            # Compute similarity between before and after
            sim = comparator.raw_similarity(png_bytes, after_bytes)
            navigated = False
            if nav_stack:
                nav_result, nav_reason = comparator.detect_navigation(nav_stack[-1][0], after_bytes)
                navigated = nav_result
            else:
                navigated = True

            print(f"      sim={sim:.4f} nav={'→' if navigated else '✗'}")

            tap_rec = TapRecord(
                index=i, label=area.label, x=ax, y=ay,
                navigated=navigated, similarity=round(sim, 4),
                after_path=after_path,
            )

            if navigated:
                parent_bytes, _ = nav_stack[-1]
                child_nav_stack = nav_stack[:-1] + [(parent_bytes, (lx, ly)), (after_bytes, None)]

                # Record child page index before recursing
                child_idx_before = len(entries)
                _dfs(phone, child_nav_stack, remaining_depth - 1, path + [page_name])

                # If a new page was created, link it
                if len(entries) > child_idx_before:
                    tap_rec.child_page_idx = child_idx_before

                print(f"    ← 返回「{page_name}」")
                _back_to(phone, nav_stack, target_name=page_name, path=path + [page_name])

            entries[idx].taps.append({
                "index": tap_rec.index,
                "label": tap_rec.label,
                "x": tap_rec.x, "y": tap_rec.y,
                "navigated": tap_rec.navigated,
                "similarity": tap_rec.similarity,
                "after_path": tap_rec.after_path,
                "child_page_idx": tap_rec.child_page_idx,
            })
            _save_dataset(out_dir, entries)

    with LivePhoneSession() as phone:
        print(f"\n开始采集: {app} (depth={depth}, sample={sample})")
        root_bytes = phone.screenshot()
        root_stack = [(root_bytes, None)]
        _dfs(phone, root_stack, depth, [])

    print(f"\n采集完成: {len(entries)} 个页面")
    print(f"输出: {out_dir}")


def _back_to(phone, nav_stack: list, target_name: str = "", path: list[str] | None = None) -> bool:
    from policy_expr.recon.back_nav import return_to_initial, manual_recover

    ok, _ = return_to_initial(
        phone.client, phone.screenshot, nav_stack,
        before_back_bytes=phone.screenshot(),
    )
    if ok:
        return True

    # Save target page screenshot for reference
    target_bytes = nav_stack[-1][0]
    path_str = " → ".join(path) if path else target_name or "上一页"
    ref_path = LOG_ROOT / "_manual_recovery_target.png"
    ref_path.write_bytes(target_bytes)

    top_level = len(nav_stack) - 1
    return manual_recover(
        phone.client, phone.screenshot, nav_stack, top_level,
        prompt=f"自动返回失败，请手动回到「{target_name}」({path_str})\n    目标截图: {ref_path}",
    )


def _save_dataset(out_dir: Path, entries: list[PageEntry]) -> None:
    data = {
        "pages": [
            {
                "idx": e.idx, "name": e.name, "dir_name": e.dir_name,
                "pairwise": e.pairwise,
                "taps": e.taps,
            }
            for e in entries
        ],
        "stats": {
            "total_pages": len(entries),
            "total_pairs": sum(len(e.pairwise) for e in entries),
            "total_taps": sum(len(e.taps) for e in entries),
            "navigated_taps": sum(1 for e in entries for t in e.taps if t["navigated"]),
        },
    }
    (out_dir / "dataset.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="截图数据采集：纯 DFS 遍历，无去重")
    ap.add_argument("app", help="应用名（如 微信）")
    ap.add_argument("--depth", type=int, default=1, help="DFS 深度 (默认 1)")
    ap.add_argument("--sample", type=int, default=0, help="每页随机采样 N 个元素 (0=全部)")
    args = ap.parse_args()
    collect(args.app, depth=args.depth, sample=args.sample)


if __name__ == "__main__":
    main()
