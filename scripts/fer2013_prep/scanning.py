"""
scanning.py

Scans the raw FER-2013 train/test folders, confirming the seven expected
emotion classes are present and each image is a 48x48 grayscale JPEG as
FER-2013 is documented to contain. Images that fail that check are
flagged rather than silently included, since a malformed image would
otherwise fail later during training instead of during data prep.
"""

from pathlib import Path

from PIL import Image

from .constants import EMOTION_CLASSES, EXPECTED_IMAGE_MODE, EXPECTED_IMAGE_SIZE, RAW_DIR


def verify_class_folders() -> None:
    """Confirm both raw splits contain exactly the seven expected emotion class folders."""
    for split in ("train", "test"):
        found = {p.name for p in (RAW_DIR / split).iterdir() if p.is_dir()}
        missing = set(EMOTION_CLASSES) - found
        unexpected = found - set(EMOTION_CLASSES)
        if missing:
            raise RuntimeError(f"raw/{split}/ is missing expected class folders: {sorted(missing)}")
        if unexpected:
            raise RuntimeError(f"raw/{split}/ has unexpected folders: {sorted(unexpected)}")


def _is_valid_image(path: Path) -> bool:
    """Check that an image matches FER-2013's documented 48x48 grayscale format."""
    try:
        with Image.open(path) as img:
            return img.size == EXPECTED_IMAGE_SIZE and img.mode == EXPECTED_IMAGE_MODE
    except OSError:
        return False


def collect_records() -> tuple[list[dict], list[str]]:
    """Scan raw/train and raw/test, returning valid records and a list of skipped filenames."""
    verify_class_folders()

    records = []
    skipped = []
    for split in ("train", "test"):
        for emotion in EMOTION_CLASSES:
            class_dir = RAW_DIR / split / emotion
            for path in sorted(class_dir.glob("*.jpg")):
                if not _is_valid_image(path):
                    skipped.append(f"{split}/{emotion}/{path.name}")
                    continue
                records.append({"path": path, "emotion": emotion, "raw_split": split})
    return records, skipped