#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_preds.py
==================
Compare saved VLSAM predictions against GT masks side-by-side.

  [ original | GT overlay (green) | predicted overlay (red) ]

Works with both flat datasets (CAMO, COD10K) and nested video folders.

Usage:
  # CAMO
  python Vision-Language-SAM/visualize_preds.py \
      -frames_root /Experiments/marcol01/CAMO/Images/Test \
      -masks_root  /Experiments/marcol01/CAMO/GT \
      -preds_root  /home/marcol01/Vision-Language-SAM/pred_masks_CAMO \
      -out_dir     /home/marcol01/viz_vlsam_CAMO

  # COD10K
  python Vision-Language-SAM/visualize_preds.py \
      -frames_root /Experiments/marcol01/COD10K-v3/Test/Image \
      -masks_root  /Experiments/marcol01/COD10K-v3/Test/GT_Object \
      -preds_root  /home/marcol01/Vision-Language-SAM/pred_masks_COD10K \
      -out_dir     /home/marcol01/viz_vlsam_COD10K

  # Nested video folders
  python Vision-Language-SAM/visualize_preds.py \
      -frames_root /Experiments/marcol01/frames \
      -masks_root  /Experiments/marcol01/masks \
      -preds_root  /home/marcol01/Vision-Language-SAM/pred_masks_blip \
      -out_dir     /home/marcol01/viz_vlsam_videos
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import json
import textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

_IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}
_MASK_EXTS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def sorted_image_paths(folder: Path) -> List[Path]:
    paths = [f for f in folder.iterdir() if f.is_file() and f.suffix in _IMG_EXTS]
    paths.sort(key=lambda f: int(f.stem) if f.stem.isdigit() else f.stem)
    return paths


def find_file(folder: Path, stem: str, exts: set) -> Optional[Path]:
    for ext in exts:
        cand = folder / (stem + ext)
        if cand.exists():
            return cand
    return None


def overlay_mask(frame_rgb: np.ndarray, mask: np.ndarray,
                 color: tuple, alpha: float = 0.5) -> np.ndarray:
    out = frame_rgb.copy().astype(np.float32)
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, dtype=np.float32)
    return out.clip(0, 255).astype(np.uint8)


def add_label(img: np.ndarray, text: str,
              color=(255, 255, 255), bg=(0, 0, 0)) -> np.ndarray:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((4, 4), text, font=font)
    draw.rectangle(bbox, fill=bg)
    draw.text((4, 4), text, fill=color, font=font)
    return np.array(pil)


def make_caption_bar(width: int, caption: str,
                     font_size: int = 18,
                     bg=(20, 20, 20), color=(240, 240, 240)) -> np.ndarray:
    """Render a full-width text bar with the BLIP caption."""
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    # Wrap text to fit the panel width
    avg_char_w = font_size * 0.6
    max_chars  = max(1, int(width / avg_char_w))
    wrapped    = "\n".join(textwrap.wrap(caption, max_chars))
    lines      = wrapped.count("\n") + 1
    bar_h      = lines * (font_size + 4) + 8
    bar        = Image.new("RGB", (width, bar_h), color=bg)
    draw       = ImageDraw.Draw(bar)
    draw.text((6, 4), wrapped, fill=color, font=font)
    return np.array(bar)


def make_panel(frame_rgb: np.ndarray,
               gt_mask: np.ndarray,
               pred_mask: np.ndarray,
               frame_name: str,
               caption: str | None = None) -> np.ndarray:
    orig_panel = add_label(frame_rgb.copy(), f"{frame_name}\noriginal")
    gt_panel   = add_label(overlay_mask(frame_rgb, gt_mask,   (0, 220,  0)), "GT mask")
    pred_panel = add_label(overlay_mask(frame_rgb, pred_mask, (220,  0,  0)), "predicted")
    row = np.concatenate([orig_panel, gt_panel, pred_panel], axis=1)
    if caption:
        bar = make_caption_bar(row.shape[1], f"BLIP: {caption}")
        row = np.concatenate([row, bar], axis=0)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Core processing
# ─────────────────────────────────────────────────────────────────────────────

def load_captions(preds_dir: Path) -> dict:
    """Load captions.json from the predictions folder if it exists."""
    p = preds_dir / "captions.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def process_folder(frames_dir: Path, masks_dir: Path, preds_dir: Path,
                   out_dir: Path, max_images: Optional[int],
                   label: str) -> None:
    """Process one flat folder of frames+GT+preds and save panels."""
    frame_paths = sorted_image_paths(frames_dir)
    if max_images:
        frame_paths = frame_paths[:max_images]

    if not frame_paths:
        print(f"[skip] No frames in {frames_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    captions = load_captions(preds_dir)
    skipped = 0

    for fp in tqdm(frame_paths, desc=label, leave=False):
        gt_path   = find_file(masks_dir, fp.stem, _MASK_EXTS)
        pred_path = find_file(preds_dir, fp.stem, {".png", ".PNG"})

        if gt_path is None:
            skipped += 1
            continue
        if pred_path is None:
            skipped += 1
            continue

        frame_pil = Image.open(fp).convert("RGB")
        H, W = frame_pil.height, frame_pil.width
        frame_rgb = np.array(frame_pil)

        gt_mask   = (np.array(Image.open(gt_path).convert("L").resize(
                        (W, H), Image.NEAREST)) > 127)
        pred_mask = (np.array(Image.open(pred_path).convert("L").resize(
                        (W, H), Image.NEAREST)) > 127)

        # Look up caption: try bare stem first, then with parent subdir
        caption = captions.get(fp.stem) or captions.get(
            str(frames_dir.name + "/" + fp.stem), None)
        panel = make_panel(frame_rgb, gt_mask, pred_mask, fp.stem, caption=caption)
        Image.fromarray(panel).save(str(out_dir / f"{fp.stem}.jpg"), quality=92)

    if skipped:
        print(f"  [{label}] skipped {skipped} frames (missing GT or prediction)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise VLSAM predictions vs GT masks side-by-side")
    p.add_argument("-frames_root", required=True,
                   help="Root folder of input frames (flat or per-video subfolders)")
    p.add_argument("-masks_root",  required=True,
                   help="Root folder of GT masks")
    p.add_argument("-preds_root",  required=True,
                   help="Root folder of saved VLSAM predictions (from Inference_vlsam.py)")
    p.add_argument("-out_dir",     default="viz_vlsam",
                   help="Output root folder for visualisation panels")
    p.add_argument("-max_images",  type=int, default=None,
                   help="Maximum images to process per folder (default: all)")
    p.add_argument("-max_videos",  type=int, default=None,
                   help="Maximum video subfolders to process (nested layout only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    frames_root = Path(args.frames_root)
    masks_root  = Path(args.masks_root)
    preds_root  = Path(args.preds_root)
    out_root    = Path(args.out_dir)

    # Auto-detect layout: flat (images directly in frames_root) vs nested
    direct_images = [f for f in frames_root.iterdir()
                     if f.is_file() and f.suffix in _IMG_EXTS]

    if direct_images:
        # ── Flat layout (CAMO, COD10K) ────────────────────────────────────
        print(f"[info] Flat layout: {len(direct_images)} images in {frames_root}")
        process_folder(
            frames_dir=frames_root,
            masks_dir=masks_root,
            preds_dir=preds_root,
            out_dir=out_root,
            max_images=args.max_images,
            label=frames_root.name,
        )
    else:
        # ── Nested layout (per-video subdirs) ─────────────────────────────
        video_dirs = sorted([d for d in frames_root.iterdir() if d.is_dir()])
        if args.max_videos:
            video_dirs = video_dirs[:args.max_videos]
        print(f"[info] Nested layout: {len(video_dirs)} video folders")

        for vd in tqdm(video_dirs, desc="videos"):
            process_folder(
                frames_dir=vd,
                masks_dir=masks_root  / vd.name,
                preds_dir=preds_root  / vd.name,
                out_dir=out_root      / vd.name,
                max_images=args.max_images,
                label=vd.name,
            )

    print(f"\nDone. Panels saved to: {out_root}")


if __name__ == "__main__":
    main()
