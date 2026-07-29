"""
parsing.py

Parses WIDER FACE's ground-truth annotation format (filename, face count,
then one "x y w h blur expression illumination invalid occlusion pose"
line per face -- with a single placeholder box line even when the count
is 0, a documented quirk of this dataset's release) into per-image
records: an image path, its pixel dimensions (needed to normalize boxes
into YOLO's 0-1 format, since the annotation file only gives absolute
pixel coordinates), and its filtered list of (x, y, w, h) boxes.
"""

from pathlib import Path

from PIL import Image

from .constants import MIN_BOX_SIZE_PX


def parse_annotation_file(gt_path: Path, images_dir: Path, source: str) -> tuple[list[dict], dict]:
    """Parse one WIDER FACE ground-truth file into per-image records plus filtering stats."""
    records = []
    stats = {"total_boxes": 0, "dropped_invalid": 0, "dropped_tiny": 0, "zero_face_images": 0}

    lines = gt_path.read_text().splitlines()
    i = 0
    while i < len(lines):
        rel_path = lines[i].strip()
        i += 1
        count = int(lines[i].strip())
        i += 1

        # WIDER FACE always emits exactly one box line, even for count == 0
        # (a placeholder "0 0 0 0 0 0 0 0 0 0" line) -- always consume it.
        raw_lines = lines[i : i + max(count, 1)]
        i += max(count, 1)

        image_path = images_dir / rel_path
        with Image.open(image_path) as img:
            width, height = img.size

        boxes = []
        if count == 0:
            stats["zero_face_images"] += 1
        else:
            for raw in raw_lines:
                x, y, w, h, _blur, _expr, _illum, invalid, _occlusion, _pose = (int(v) for v in raw.split())
                stats["total_boxes"] += 1
                if invalid == 1:
                    stats["dropped_invalid"] += 1
                    continue
                if w < MIN_BOX_SIZE_PX or h < MIN_BOX_SIZE_PX:
                    stats["dropped_tiny"] += 1
                    continue
                boxes.append((x, y, w, h))

        records.append(
            {
                "path": image_path,
                "unique_name": f"{source}__{rel_path.replace('/', '__')}",
                "width": width,
                "height": height,
                "boxes": boxes,
            }
        )

    return records, stats