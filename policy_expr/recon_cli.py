"""CLI entry point for app reconnaissance: parse page structure from screenshots.

Usage:
    # Online: capture phone screen, parse, and probe elements
    uv run python -m policy_expr.recon_cli

    # Offline: analyze existing image(s)
    uv run python -m policy_expr.recon_cli images/recon/
    uv run python -m policy_expr.recon_cli images/recon/policy_xxx.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from policy_expr.recon import PageParser, viz_result
from policy_expr.recon.bfs import probe_elements

ROOT = Path(__file__).parent.parent
LOG_ROOT = ROOT / "logs" / "recon"


def run_online() -> None:
    """Capture phone screen, parse, and probe each element."""
    from policy_expr.perception import LivePhoneSession

    print("在线模式: 截取手机屏幕...")
    with LivePhoneSession() as phone:
        out_dir = LOG_ROOT / "online"
        out_dir.mkdir(parents=True, exist_ok=True)

        png_bytes = phone.screenshot()
        img_path = out_dir / "screenshot.png"
        img_path.write_bytes(png_bytes)
        print(f"截图已保存: {img_path}")

        print("解析中 (LLM + YOLO + merge)...")
        page = PageParser().parse_screen(png_bytes)
        viz_result(page, png_bytes, "screenshot", out_dir)

        result = probe_elements(phone.client, page, out_dir)
        result_path = out_dir / "recon_result.json"
        result.save(result_path)

        print(f"\n{'=' * 60}")
        print(f"探测完成: {len(result.taps)} 个元素")
        ok = sum(1 for t in result.taps if t.tap_ok)
        print(f"  成功: {ok}  失败: {len(result.taps) - ok}")
        print(f"  结果: {result_path}")
        print(f"{'=' * 60}")


def run_offline(paths: list[Path]) -> None:
    """Analyze existing image files."""
    images: list[Path] = []
    for p in paths:
        if p.is_file() and p.suffix.lower() == ".png":
            images.append(p)
        elif p.is_dir():
            images.extend(sorted(p.glob("*.png")))

    if not images:
        print("未找到 PNG 图片")
        return

    print(f"离线模式: 共 {len(images)} 张图片")
    parser = PageParser()
    out_dir = LOG_ROOT / "offline"
    for img_path in images:
        png_bytes = img_path.read_bytes()
        print(f"\n{'=' * 60}")
        print(f"图片: {img_path.name}")
        print("解析中 (LLM + YOLO + merge)...")
        page = parser.parse_screen(png_bytes)
        viz_result(page, png_bytes, img_path.stem, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="App Recon: 页面结构分析")
    ap.add_argument(
        "paths", nargs="*", type=Path,
        help="图片文件或目录（留空则在线截图+探测）",
    )
    args = ap.parse_args()

    if args.paths:
        run_offline(args.paths)
    else:
        run_online()


if __name__ == "__main__":
    main()
