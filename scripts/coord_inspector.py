"""
coord_inspector.py — 截图坐标查看器

用法:
    uv run python scripts/coord_inspector.py [image_path]

- 鼠标悬停：实时显示 0-1000 归一化坐标
- 左键点击：打印坐标到终端，并复制到剪贴板
- 滚轮 / +- 键：缩放
- 拖动（中键或空格+左键）：平移
"""
from __future__ import annotations

import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

from PIL import Image, ImageTk


# ── constants ────────────────────────────────────────────────────────────────
ZOOM_STEP = 1.15
MIN_ZOOM  = 0.1
MAX_ZOOM  = 20.0


def copy_to_clipboard(text: str) -> None:
    subprocess.run(["pbcopy"], input=text.encode(), check=False)


class CoordInspector:
    def __init__(self, root: tk.Tk, image_path: Path) -> None:
        self.root = root
        self.root.title(f"Coord Inspector — {image_path.name}")

        self.img_orig = Image.open(image_path).convert("RGB")
        self.iw, self.ih = self.img_orig.size

        # canvas fills window
        self.canvas = tk.Canvas(root, cursor="crosshair", bg="#1e1e1e",
                                highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # status bar
        self.status = tk.Label(root, text="", anchor="w",
                               font=("Menlo", 13), bg="#2d2d2d", fg="#e0e0e0",
                               padx=8, pady=4)
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

        # state
        self.zoom   = 1.0
        self.offset: list[float] = [0.0, 0.0]   # canvas offset (pixels) of image top-left
        self._drag_start: tuple[int, int] | None = None
        self._space_held = False
        self._photo: ImageTk.PhotoImage | None = None

        root.update_idletasks()
        self._fit_to_window()
        self._redraw()

        # bindings
        self.canvas.bind("<Motion>",          self._on_motion)
        self.canvas.bind("<ButtonPress-1>",   self._on_press1)
        self.canvas.bind("<B1-Motion>",       self._on_drag1)
        self.canvas.bind("<ButtonPress-2>",   self._drag_start_cb)
        self.canvas.bind("<B2-Motion>",       self._drag_move_cb)
        self.canvas.bind("<MouseWheel>",      self._on_scroll_mac)
        self.canvas.bind("<Button-4>",        self._on_scroll_linux)
        self.canvas.bind("<Button-5>",        self._on_scroll_linux)
        root.bind("<KeyPress-space>",         lambda _: setattr(self, "_space_held", True))
        root.bind("<KeyRelease-space>",       lambda _: setattr(self, "_space_held", False))
        root.bind("<plus>",  lambda _: self._zoom_by(ZOOM_STEP))
        root.bind("<equal>", lambda _: self._zoom_by(ZOOM_STEP))
        root.bind("<minus>", lambda _: self._zoom_by(1 / ZOOM_STEP))
        root.bind("<Configure>", lambda _: self._redraw())

    # ── coordinate helpers ───────────────────────────────────────────────────

    def _canvas_to_norm(self, cx: int, cy: int) -> tuple[float, float] | None:
        """Canvas pixel → 0-1000 normalized coordinates."""
        ix = (cx - self.offset[0]) / self.zoom
        iy = (cy - self.offset[1]) / self.zoom
        if 0 <= ix <= self.iw and 0 <= iy <= self.ih:
            return ix / self.iw * 1000, iy / self.ih * 1000
        return None

    # ── drawing ──────────────────────────────────────────────────────────────

    def _fit_to_window(self) -> None:
        cw = self.canvas.winfo_width()  or 800
        ch = self.canvas.winfo_height() or 600
        self.zoom = min(cw / self.iw, ch / self.ih)
        self.offset = [
            (cw - self.iw * self.zoom) / 2,
            (ch - self.ih * self.zoom) / 2,
        ]

    def _redraw(self) -> None:
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        nw = max(1, int(self.iw * self.zoom))
        nh = max(1, int(self.ih * self.zoom))
        scaled = self.img_orig.resize((nw, nh), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(scaled)
        self.canvas.delete("all")
        self.canvas.create_image(self.offset[0], self.offset[1],
                                  anchor="nw", image=self._photo)

    # ── events ───────────────────────────────────────────────────────────────

    def _on_motion(self, event: tk.Event) -> None:
        coord = self._canvas_to_norm(event.x, event.y)
        if coord:
            x, y = coord
            self.status.config(
                text=f"  ({x:6.1f}, {y:6.1f})   zoom {self.zoom:.2f}×   "
                     f"[点击复制]"
            )
        else:
            self.status.config(text="  —")

    def _on_press1(self, event: tk.Event) -> None:
        if self._space_held:
            self._drag_start = (event.x, event.y)
            return
        coord = self._canvas_to_norm(event.x, event.y)
        if coord:
            x, y = coord
            text = f"({x:.1f}, {y:.1f})"
            copy_to_clipboard(text)
            print(f"[coord] {text}")
            self.status.config(text=f"  复制: {text}   ✓")

    def _on_scroll_mac(self, event: tk.Event) -> None:
        factor = ZOOM_STEP if event.delta > 0 else 1 / ZOOM_STEP
        self._zoom_at(factor, event.x, event.y)

    def _on_scroll_linux(self, event: tk.Event) -> None:
        factor = ZOOM_STEP if event.num == 4 else 1 / ZOOM_STEP
        self._zoom_at(factor, event.x, event.y)

    def _zoom_at(self, factor: float, cx: int, cy: int) -> None:
        new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom * factor))
        ratio = new_zoom / self.zoom
        self.offset[0] = cx - ratio * (cx - self.offset[0])
        self.offset[1] = cy - ratio * (cy - self.offset[1])
        self.zoom = new_zoom
        self._redraw()

    def _zoom_by(self, factor: float) -> None:
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        self._zoom_at(factor, cw // 2, ch // 2)

    def _drag_start_cb(self, event: tk.Event) -> None:
        self._drag_start = (event.x, event.y)

    def _drag_move_cb(self, event: tk.Event) -> None:
        if self._drag_start:
            dx = event.x - self._drag_start[0]
            dy = event.y - self._drag_start[1]
            self.offset[0] += dx
            self.offset[1] += dy
            self._drag_start = (event.x, event.y)
            self._redraw()

    def _on_drag1(self, event: tk.Event) -> None:
        if self._space_held and self._drag_start:
            dx = event.x - self._drag_start[0]
            dy = event.y - self._drag_start[1]
            self.offset[0] += dx
            self.offset[1] += dy
            self._drag_start = (event.x, event.y)
            self._redraw()


def main() -> None:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        root_tmp = tk.Tk()
        root_tmp.withdraw()
        p = filedialog.askopenfilename(
            title="选择截图",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp"), ("All", "*.*")],
        )
        root_tmp.destroy()
        if not p:
            sys.exit(0)
        path = Path(p)

    if not path.exists():
        print(f"文件不存在: {path}", file=sys.stderr)
        sys.exit(1)

    root = tk.Tk()
    root.geometry("900x700")
    CoordInspector(root, path)
    root.mainloop()


if __name__ == "__main__":
    main()
