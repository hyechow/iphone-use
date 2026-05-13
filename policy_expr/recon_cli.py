"""CLI entry point for app reconnaissance and self-learning.

Usage:
    # Online: capture phone screen, parse, and probe elements
    uv run python -m policy_expr.recon_cli

    # Offline: analyze existing image(s)
    uv run python -m policy_expr.recon_cli images/recon/
    uv run python -m policy_expr.recon_cli images/recon/policy_xxx.png

    # Self-learning: build page flows from recon result
    uv run python -m policy_expr.recon_cli --learn logs/recon/online_x/recon_result.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from PIL import Image

load_dotenv()

from policy_expr.recon import PageParser, viz_result
from policy_expr.recon.bfs import probe_elements

ROOT = Path(__file__).parent.parent
LOG_ROOT = ROOT / "logs" / "recon"
KNOWLEDGE_ROOT = ROOT / "knowledge"
OFFLINE_EXPECTED_SIZE = (636, 1402)


def _parse_frontmatter(path: Path) -> dict[str, str]:
    """Parse YAML frontmatter from a markdown file."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}
    end = text.find("---", 3)
    if end < 0:
        return {}
    meta = {}
    for line in text[3:end].strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta


def _check_knowledge_exists(app: str, signature: str) -> Path | None:
    """Check if a page with given signature already exists in knowledge.

    Returns the matching .md path if found, None otherwise.
    """
    app_dir = KNOWLEDGE_ROOT / app
    if not app_dir.is_dir():
        return None

    for md_path in sorted(app_dir.glob("*.md")):
        meta = _parse_frontmatter(md_path)
        if meta.get("signature") == signature:
            return md_path
    return None


def run_app(app: str, *, debug: bool = False) -> None:
    """Online recon for a specific app with knowledge dedup check."""
    from policy_expr.perception import LivePhoneSession

    print(f"应用侦察: {app}")

    # Step 1: screenshot + parse identity
    with LivePhoneSession() as phone:
        png_bytes = phone.screenshot()

        print("解析页面身份...")
        knowledge = PageParser().analyze_screen(png_bytes)
        ident = knowledge.page.identity
        signature = ident.signature
        # Derive stable page name from signature (2nd segment), not page_title
        # signature format: "app/page_name/stable_features"
        sig_parts = signature.split("/")
        page_name = sig_parts[1] if len(sig_parts) >= 2 else ident.page_title

        print(f"  应用: {ident.app_name}")
        print(f"  页面: {page_name}")
        print(f"  签名: {signature}")

    # Step 2: check if already learned
    existing = _check_knowledge_exists(app, signature)
    if existing:
        print(f"\n已学习过此页面: {existing}")
        print(f"签名: {signature}")
        return

    print("新页面，开始探测...")

    # Step 3: full recon
    with LivePhoneSession() as phone:
        out_dir = LOG_ROOT / app / page_name
        out_dir.mkdir(parents=True, exist_ok=True)

        img_path = out_dir / "initial.png"
        img_path.write_bytes(png_bytes)
        print(f"截图已保存: {img_path}")

        viz_result(knowledge, png_bytes, "initial", out_dir)

        result = probe_elements(
            phone.client, knowledge, out_dir, img_path, phone.screenshot,
            debug=debug,
        )
        result_path = out_dir / "recon_result.json"
        result.save(result_path)

        print(f"\n{'=' * 60}")
        print(f"探测完成: {len(result.taps)} 个元素")
        ok = sum(1 for t in result.taps if t.tap_ok)
        print(f"  成功: {ok}  失败: {len(result.taps) - ok}")
        print(f"  结果: {out_dir}")
        print(f"{'=' * 60}")
    """Capture phone screen, parse, and probe each element."""
    from policy_expr.perception import LivePhoneSession

    print("在线模式: 截取手机屏幕...")
    with LivePhoneSession() as phone:
        out_dir = LOG_ROOT / "online"
        out_dir.mkdir(parents=True, exist_ok=True)

        png_bytes = phone.screenshot()
        img_path = out_dir / "initial.png"
        img_path.write_bytes(png_bytes)
        print(f"截图已保存: {img_path}")

        print("解析中 (LLM + YOLO + merge + 分组)...")
        knowledge = PageParser().analyze_screen(png_bytes)
        viz_result(knowledge, png_bytes, "initial", out_dir)

        result = probe_elements(
            phone.client, knowledge, out_dir, img_path, phone.screenshot,
            debug=debug,
        )
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
        with Image.open(img_path) as img:
            if img.size != OFFLINE_EXPECTED_SIZE:
                print(
                    f"离线测试截图尺寸不匹配: {img_path} "
                    f"实际 {img.width}x{img.height}，"
                    f"期望 {OFFLINE_EXPECTED_SIZE[0]}x{OFFLINE_EXPECTED_SIZE[1]}"
                )
                raise SystemExit(1)
        initial_path = out_dir / f"{img_path.stem}_initial.png"
        initial_path.parent.mkdir(parents=True, exist_ok=True)
        initial_path.write_bytes(png_bytes)
        print(f"\n{'=' * 60}")
        print(f"图片: {img_path.name}")
        print(f"初始截图: {initial_path}")
        print("解析中 (LLM + YOLO + merge + 分组)...")
        knowledge = parser.analyze_screen(png_bytes)
        viz_result(knowledge, png_bytes, img_path.stem, out_dir)


def run_learn(recon_path: Path) -> None:
    """Build page flow descriptions from a recon result."""
    from policy_expr.self_learning.flow import build_page_flows, save_page_flows

    print(f"自学习模式: 从侦察结果生成功能描述...")
    print(f"输入: {recon_path}")

    page_flow = build_page_flows(recon_path)

    out_path = recon_path.parent / "page_flows.json"
    save_page_flows(page_flow, out_path)

    print(f"\n{'=' * 60}")
    print(f"功能流程: {len(page_flow.flows)} 条")
    for f in page_flow.flows:
        print(f"  {f.flow_description}")
    print(f"结果: {out_path}")
    print(f"{'=' * 60}")


def _resolve_knowledge_path(recon_dir: Path) -> Path:
    """Derive knowledge output path from recon directory.

    logs/recon/{app}/{page} → knowledge/{app}/{page}.md
    """
    # Try to extract app/page from path relative to logs/recon
    try:
        relative = recon_dir.resolve().relative_to(LOG_ROOT.resolve())
        parts = relative.parts  # e.g. ('微信', '主界面')
        if len(parts) >= 2:
            app, page = parts[0], parts[1]
        else:
            # Fallback: use recon dir name as page, read app from result
            app = "unknown"
            page = parts[0] if parts else recon_dir.name
    except ValueError:
        app = "unknown"
        page = recon_dir.name

    return ROOT / "knowledge" / app / f"{page}.md"


def run_knowledge(recon_dir: Path) -> None:
    """Build page operation knowledge base from recon results."""
    from policy_expr.self_learning.knowledge import build_knowledge, save_knowledge

    print(f"知识库构建: 从侦察结果生成页面操作知识...")
    print(f"输入: {recon_dir}")

    kb = build_knowledge(recon_dir)
    out_path = _resolve_knowledge_path(recon_dir)

    # Also save a copy in recon dir for reference
    save_knowledge(kb, recon_dir / "knowledge.md")

    # Save to knowledge directory
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_knowledge(kb, out_path)

    print(f"\n{'=' * 60}")
    print(kb.to_skill())
    print(f"\n知识库: {out_path}")
    print(f"{'=' * 60}")


def main() -> None:
    ap = argparse.ArgumentParser(description="App Recon: 页面结构分析与自学习")
    ap.add_argument(
        "paths", nargs="*", type=Path,
        help="图片文件或目录（留空则在线截图+探测）",
    )
    ap.add_argument("--debug", action="store_true", help="调试模式，每个元素暂停")
    ap.add_argument("--app", type=str, metavar="NAME", help="在线侦察指定应用，自动去重")
    ap.add_argument("--learn", type=Path, metavar="JSON", help="从侦察结果生成功能流程描述")
    ap.add_argument("--knowledge", type=Path, metavar="DIR", help="从侦察目录构建页面操作知识库")
    args = ap.parse_args()

    if args.app:
        run_app(args.app, debug=args.debug)
    elif args.learn:
        run_learn(args.learn)
    elif args.knowledge:
        run_knowledge(args.knowledge)
    elif args.paths:
        run_offline(args.paths)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
