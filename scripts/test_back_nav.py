"""Back-navigation test: script-driven multi-level navigation + return_to_initial.

Usage:
    uv run python scripts/test_back_nav.py

Flow:
    1. Connect to phone, take root screenshot.
    2. Render annotated image with numbered tap targets; open it automatically.
    3. User types a number → script taps that element, confirms navigation.
    4. Repeat for desired depth (q to stop and start back-nav test).
    5. Call return_to_initial; print result.
"""

from __future__ import annotations

import io
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from PIL import Image, ImageDraw, ImageFont

from policy_expr.perception import LivePhoneSession
from policy_expr.executor import logical_xy
from policy_expr.recon.page_compare import make_comparator
from policy_expr.recon.page_parser import PageParser, classify_elements
from policy_expr.recon.back_nav import return_to_initial, BACK_SETTLE_SECONDS

MAX_DEPTH = 5
SETTLE = BACK_SETTLE_SECONDS

# Colour palette for numbered badges (R, G, B)
_PALETTE = [
    (255, 59, 48),   (255, 149, 0),  (255, 204, 0),  (52, 199, 89),
    (0, 199, 190),   (50, 173, 230), (0, 122, 255),  (88, 86, 214),
    (175, 82, 222),  (255, 45, 85),
]


# ── Visualization ─────────────────────────────────────────────────────────────

def annotate(png: bytes, items: list[tuple[float, float, str]]) -> bytes:
    """Draw numbered circles + label chips on screenshot; return annotated PNG."""
    img = Image.open(io.BytesIO(png)).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    # Try to load a readable font; fall back to default
    try:
        font_num = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=22)
    except Exception:
        font_num = ImageFont.load_default()

    R = 18  # badge radius

    for i, item in enumerate(items):
        ax, ay = item[0], item[1]
        cx = int(ax / 1000 * w)
        cy = int(ay / 1000 * h)
        color = _PALETTE[i % len(_PALETTE)]

        # Filled circle badge
        draw.ellipse(
            [cx - R, cy - R, cx + R, cy + R],
            fill=(*color, 230),
            outline=(255, 255, 255, 255),
            width=2,
        )

        # Number centred in badge
        num_str = str(i)
        bb = draw.textbbox((0, 0), num_str, font=font_num)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        draw.text((cx - tw / 2, cy - th / 2 - 1), num_str,
                  fill=(255, 255, 255), font=font_num)


    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def show_and_open(png: bytes, items: list[tuple[float, float, str]],
                  depth: int) -> Path:
    """Save annotated image to /tmp and open it in Preview."""
    annotated = annotate(png, items)
    out = Path(tempfile.gettempdir()) / f"back_nav_depth{depth}.png"
    out.write_bytes(annotated)
    subprocess.Popen(["open", str(out)])
    return out


# ── Element listing ───────────────────────────────────────────────────────────

def parse_items(png: bytes, parser: PageParser) -> list[tuple[float, float, str]]:
    areas = classify_elements(parser.parse_screen(png))
    return [(a.center_xy[0], a.center_xy[1], a.label[:30] or "(无标签)")
            for a in areas]


def print_items(items: list[tuple[float, float, str]]) -> None:
    print(f"\n  {'#':>3}  {'坐标':^12}  标签")
    print(f"  {'-'*3}  {'-'*12}  {'-'*24}")
    for i, (ax, ay, label) in enumerate(items):
        print(f"  {i:>3}  ({ax:>5.0f},{ay:>4.0f})  {label}")


def prompt_choice(n: int) -> int | None:
    while True:
        raw = input("\n  编号（q=结束导航 / 开始回退测试）: ").strip()
        if raw.lower() == "q":
            return None
        if raw.isdigit() and 0 <= int(raw) < n:
            return int(raw)
        print(f"  请输入 0–{n-1} 之间的数字")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = PageParser()
    change_comp = make_comparator("edge_iou")

    with LivePhoneSession() as phone:
        screenshot = phone.screenshot
        assert phone.client is not None

        # Root
        print("\n" + "="*60)
        print("  [root] 截取当前页面作为回退目标")
        print("="*60)
        root_png = screenshot()
        nav_stack: list[tuple[bytes, tuple[float, float] | None]] = [(root_png, None)]
        # tap_labels[i] = label of element tapped to go from L_i → L_{i+1}
        tap_labels: list[str] = []
        current_png = root_png

        # Navigate down
        for depth in range(1, MAX_DEPTH + 1):
            print(f"\n{'='*60}")
            print(f"  [depth {depth}] 解析页面元素")
            print("="*60)

            items = parse_items(current_png, parser)
            if not items:
                print("  未解析到可交互元素，停止")
                break

            img_path = show_and_open(current_png, items, depth)
            print(f"  已标注截图: {img_path}")
            print_items(items)

            choice = prompt_choice(len(items))
            if choice is None:
                print("  开始回退测试")
                break

            ax, ay = items[choice][0], items[choice][1]
            label = items[choice][2]
            lx, ly = logical_xy(ax, ay)

            print(f"\n  → tap [{label}] at ({ax:.0f}, {ay:.0f})")
            phone.client.tap(lx, ly)
            time.sleep(SETTLE)
            after_png = screenshot()

            unchanged, sim = change_comp.no_change_score(current_png, after_png)
            if unchanged:
                print(f"  页面未变化 (edge_iou={sim:.3f})，请重选")
                continue

            print(f"  ✓ 导航成功 (edge_iou={sim:.3f})")
            prev_png, _ = nav_stack[-1]
            nav_stack[-1] = (prev_png, (lx, ly))
            nav_stack.append((after_png, None))
            tap_labels.append(label)
            current_png = after_png

            if depth == MAX_DEPTH:
                print(f"\n  已达最大深度 {MAX_DEPTH}")

        # Back-nav test
        depth_reached = len(nav_stack) - 1
        if depth_reached == 0:
            print("\n  未曾导航，退出")
            return

        # Ask how many levels to go back (default 1)
        while True:
            raw = input(f"\n  回退几层？(1–{depth_reached}，默认 1): ").strip()
            if raw == "":
                back_n = 1
                break
            if raw.isdigit() and 1 <= int(raw) <= depth_reached:
                back_n = int(raw)
                break
            print(f"  请输入 1–{depth_reached} 之间的数字")

        # nav_stack[-1] is the target page for return_to_initial.
        # To go back n levels: target = nav_stack[-n-1], so pass nav_stack[:-n].
        target_stack = nav_stack[:-back_n]

        # nav_context = how we got FROM the target level INTO the next level
        # = tap_labels[target_level], where target_level = depth_reached - back_n
        target_level = depth_reached - back_n
        nav_context = tap_labels[target_level] if target_level < len(tap_labels) else ""

        print(f"\n{'='*60}")
        print(f"  [回退测试] 当前深度 L{depth_reached}，回退 {back_n} 层 → L{target_level}")
        if nav_context:
            print(f"  nav_context: 「{nav_context}」")
        print("="*60)
        for i in range(len(target_stack)):
            coords = target_stack[i][1]
            target_mark = " ← 目标" if i == len(target_stack) - 1 else ""
            print(f"  L{i}: {'→ ' + str(coords) if coords else '(无 forward 坐标)'}{target_mark}")

        # Prepare debug output dir
        import json, tempfile, datetime
        out_dir = Path(tempfile.gettempdir()) / f"back_nav_{datetime.datetime.now():%H%M%S}"
        out_dir.mkdir()
        print(f"  调试目录: {out_dir}")

        # Save nav_stack reference screenshots so we can see what each level looks like
        for i in range(len(target_stack)):
            ref_png = target_stack[i][0]
            ref_path = out_dir / f"ref_L{i}{'_target' if i == len(target_stack)-1 else ''}.png"
            ref_path.write_bytes(ref_png)
        print(f"  参考截图: ref_L0.png … ref_L{len(target_stack)-1}_target.png")

        print()
        before_back = screenshot()
        success, log = return_to_initial(
            client=phone.client,
            screenshot=screenshot,
            nav_stack=target_stack,
            before_back_bytes=before_back,
            out_dir=out_dir,
            nav_context=nav_context,
        )

        # Save structured log
        log_data = {
            "success": success,
            "depth_reached": depth_reached,
            "back_n": back_n,
            "target_level": depth_reached - back_n,
            "nav_context": nav_context,
            "steps": log,
        }
        log_path = out_dir / "log.json"
        log_path.write_text(json.dumps(log_data, ensure_ascii=False, indent=2))
        print(f"  日志已保存: {log_path}")

        print(f"\n{'='*60}")
        target_level = depth_reached - back_n
        print(f"  结果: {'✓ 成功回到 L' + str(target_level) if success else '✗ 未能回到 L' + str(target_level)}")
        print("="*60)
        for i, entry in enumerate(log):
            s = entry.get("strategy", "?")
            r = entry.get("result", "")
            score = entry.get("score", "")
            score_str = f" ({score:.3f})" if isinstance(score, float) else ""
            llm_method = entry.get("llm_method", "")
            llm_str = f"  [LLM: {llm_method}]" if llm_method else ""
            mark = "✓" if entry.get("success") else "·"
            print(f"  {mark} [{i+1:02d}] {s:12s} → {r}{score_str}{llm_str}")


if __name__ == "__main__":
    main()
