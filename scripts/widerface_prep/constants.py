"""
constants.py

Shared paths and preparation parameters for WIDER FACE preparation: where
the raw source images/annotations and processed output live, the box-size
filtering threshold, and the split parameters.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

RAW_DIR = REPO_ROOT / "data" / "widerface" / "raw"
RAW_TRAIN_IMAGES_DIR = RAW_DIR / "WIDER_train" / "images"
RAW_VAL_IMAGES_DIR = RAW_DIR / "WIDER_val" / "images"
TRAIN_GT_PATH = RAW_DIR / "wider_face_split" / "wider_face_train_bbx_gt.txt"
VAL_GT_PATH = RAW_DIR / "wider_face_split" / "wider_face_val_bbx_gt.txt"

PROCESSED_DIR = REPO_ROOT / "data" / "widerface" / "processed"
REPORT_PATH = PROCESSED_DIR / "distribution_report.json"
DATASET_YAML_PATH = PROCESSED_DIR / "widerface.yaml"

CLASS_NAMES = ["face"]

# Boxes narrower or shorter than this (pixels, in the original image) are
# dropped as unusable noise rather than real training signal -- WIDER FACE's
# crowd-scene photography produces many near-degenerate annotations (down to
# 0px) that a retail single-subject detector has no use for. See
# docs/datasets/widerface.md for the measured size distribution behind this
# choice.
MIN_BOX_SIZE_PX = 8

# The official WIDER FACE test set ships with no public ground truth, so it
# isn't usable for training or evaluation. Instead, the official train split
# is kept as-is, and the official val split (which does have ground truth)
# is divided into our own val and test sets.
VAL_TEST_SPLIT_RATIO = 0.5  # share of official val assigned to our val split; remainder becomes test
RANDOM_SEED = 42

# Buckets used to stratify the val/test split so neither ends up skewed
# toward sparse or crowd-dense images relative to the other.
FACE_COUNT_BUCKETS = [
    (0, 0, "0"),
    (1, 1, "1"),
    (2, 5, "2-5"),
    (6, 20, "6-20"),
    (21, 50, "21-50"),
    (51, 100, "51-100"),
    (101, float("inf"), "101+"),
]