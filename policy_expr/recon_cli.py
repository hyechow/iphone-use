"""CLI entry point for app reconnaissance: parse page structure from screenshots.

Usage:
    # Online: capture phone screen and analyze
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

from policy_expr.recon import PageParser, print_result, visualize

ROOT = Path(__file__).parent.parent
LOG_ROOT = ROOT / "logs" / "recon"


def process_image(parser: PageParser, img_path: Path, out_dir: Path) -> None:
    """Parse one image: run pipeline, save JSON + viz, print result."""
    png_bytes = img_path.read_bytes()
    print(f"\n{'=' * 60}")
    print(f"图片: {img_path.name}")

    print("解析中 (LLM + YOLO + merge)...")
    page = parser.parse_screen(png_bytes)
    print_result(page)

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{img_path.stem}_result.json"
    json_path.write_text(page.model_dump_json(indent=2), encoding="utf-8")

    viz_path = out_dir / f"{img_path.stem}_viz.png"
    viz_path.write_bytes(visualize(page, png_bytes))

    print(f"\n  JSON : {json_path}")
    print(f"  可视化 : {viz_path}")


def run_online() -> None:
    """Capture phone screen and analyze."""
    from policy_expr.perception import LivePhoneSession

    print("在线模式: 截取手机屏幕...")
    with LivePhoneSession() as phone:
        png_bytes = phone.screenshot()

    out_dir = LOG_ROOT / "online"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_path = out_dir / "screenshot.png"
    img_path.write_bytes(png_bytes)
    print(f"截图已保存: {img_path}")

    process_image(PageParser(), img_path, out_dir)


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
        process_image(parser, img_path, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="App Recon: 页面结构分析")
    ap.add_argument(
        "paths", nargs="*", type=Path,
        help="图片文件或目录（留空则在线截图）",
    )
    args = ap.parse_args()

    if args.paths:
        run_offline(args.paths)
    else:
        run_online()


if __name__ == "__main__":
    main()
