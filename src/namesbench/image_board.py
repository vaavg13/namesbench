from __future__ import annotations

import math
import random
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


@dataclass
class BoardImages:
    composite_path: Path
    composite_data_url: str
    index_to_file: Dict[int, Path]
    index_to_team: Dict[int, str]
    debug_path: Optional[Path] = None


def build_image_board(
    deck_path: Path,
    grid_size: Tuple[int, int],
    output_dir: Path,
    friendly_fraction: float = 0.5,
    seed: Optional[int] = None,
    debug: bool = False,
) -> BoardImages:
    if not deck_path.exists() or not deck_path.is_dir():
        raise FileNotFoundError(f"Deck directory not found: {deck_path}")

    images = sorted([p for p in deck_path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not images:
        raise RuntimeError(f"No images found in deck: {deck_path}")

    rows, cols = grid_size
    total_slots = rows * cols

    rng = random.Random(seed)
    sampled = rng.sample(images, k=total_slots)

    friendly_count = max(1, math.ceil(total_slots * friendly_fraction))
    friendly_indices = set(rng.sample(range(total_slots), k=friendly_count))

    index_to_team: Dict[int, str] = {}
    for idx in range(total_slots):
        index_to_team[idx + 1] = "friendly" if idx in friendly_indices else "opponent"

    index_to_file: Dict[int, Path] = {idx + 1: sampled[idx] for idx in range(total_slots)}

    output_dir.mkdir(parents=True, exist_ok=True)
    composite_path = (output_dir / "board_composite.png").resolve()
    debug_path = (output_dir / "board_debug.png").resolve()

    _compose_board_image(index_to_file, rows, cols, composite_path)
    composite_data_url = _image_to_data_url(composite_path)
    final_debug_path = None
    if debug:
        final_debug_path = _compose_board_image(index_to_file, rows, cols, debug_path, index_to_team)

    return BoardImages(
        composite_path=composite_path,
        composite_data_url=composite_data_url,
        index_to_file=index_to_file,
        index_to_team=index_to_team,
        debug_path=final_debug_path,
    )


def _compose_board_image(
    index_to_file: Dict[int, Path],
    rows: int,
    cols: int,
    output_path: Path,
    index_to_team: Optional[Dict[int, str]] = None,
) -> Path:
    images: List[Image.Image] = []
    max_width = 0
    max_height = 0
    for path in index_to_file.values():
        img = Image.open(path).convert("RGB")
        images.append(img)
        max_width = max(max_width, img.width)
        max_height = max(max_height, img.height)

    board_width = cols * max_width
    board_height = rows * max_height
    board = Image.new("RGB", (board_width, board_height), color=(30, 30, 30))
    draw = ImageDraw.Draw(board)

    font_size = max(48, int(min(max_width, max_height) * 0.18))
    font = _load_font(font_size)

    for (index, img) in zip(index_to_file.keys(), images):
        idx0 = index - 1
        row = idx0 // cols
        col = idx0 % cols
        x_offset = col * max_width
        y_offset = row * max_height

        resized = img.resize((max_width, max_height))
        board.paste(resized, (x_offset, y_offset))

        number_bg = (0, 0, 0)
        label = str(index)
        # Pillow >=10 removed textsize; use textbbox for robust size calculation
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        padding = max(10, font_size // 8)
        rect = [
            x_offset + max_width - text_width - 2 * padding,
            y_offset + max_height - text_height - 2 * padding,
            x_offset + max_width,
            y_offset + max_height,
        ]
        draw.rectangle(rect, fill=number_bg)
        draw.text((rect[0] + padding, rect[1] + padding), label, font=font, fill=(255, 255, 255))

        if index_to_team:
            team = index_to_team.get(index)
            if team == "friendly":
                outline = (0, 200, 0)
            else:
                outline = (200, 0, 0)
            draw.rectangle(
                [x_offset, y_offset, x_offset + max_width, y_offset + max_height],
                outline=outline,
                width=6,
            )

    board.save(output_path)
    for img in images:
        img.close()
    return output_path


def _load_font(font_size: int) -> ImageFont.ImageFont:
    candidates = [
        "DejaVuSans-Bold.ttf",
        "Arial.ttf",
        "LiberationSans-Bold.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def _image_to_data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


