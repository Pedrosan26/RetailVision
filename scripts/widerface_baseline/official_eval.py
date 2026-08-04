"""
official_eval.py

Computes recall against WIDER FACE's real, official Easy/Medium/Hard
ground-truth partition (data/widerface/raw/wider_face_split/ground_truth/
wider_{easy,medium,hard}_val.mat -- downloaded from the dataset's
eval_tools package, http://shuoyang1213.me/WIDERFACE/support/eval_script/
eval_tools.zip). Each .mat file lists, per official-val image, which face
indices count toward that difficulty level -- the partition itself was
computed by the dataset's original authors via a proposal-recall
procedure, not a simple box-size threshold, so it can't be reconstructed
from the raw annotations; it has to be read from these files directly.

This is evaluated only against images that fall in *our* test split
(1,613 of the official val set's 3,226 images -- the rest went to our
val split, see widerface_prep.splitting), using the .mat files' own
face_bbx_list as ground truth rather than our own already-filtered
widerface_prep records, so this stays faithful to the official protocol
rather than mixing in our own preprocessing choices.

Recall (not full precision/AP) is computed via greedy IoU>=0.5 matching
between each image's official keep-boxes and the model's predictions --
a direct, simpler read of "how many of the official ground-truth faces
did the model find" than reproducing WIDER FACE's full confidence-ranked
AP curve algorithm, which the "recall documented" acceptance criterion
doesn't require.
"""

from pathlib import Path

import scipy.io as sio
from ultralytics import YOLO

from widerface_prep.constants import RAW_DIR, RAW_VAL_IMAGES_DIR

GROUND_TRUTH_DIR = RAW_DIR / "wider_face_split" / "ground_truth"
IOU_THRESHOLD = 0.5


def _load_keep_boxes(mat_path: Path) -> dict[str, list[tuple[int, int, int, int]]]:
    """Parse one difficulty's .mat file into {event/filename.jpg: [keep boxes]}."""
    data = sio.loadmat(str(mat_path))
    event_list, file_list, bbx_list, gt_list = (
        data["event_list"],
        data["file_list"],
        data["face_bbx_list"],
        data["gt_list"],
    )

    keep_boxes: dict[str, list[tuple[int, int, int, int]]] = {}
    for event_idx in range(len(event_list)):
        event_name = str(event_list[event_idx][0][0])
        event_files = file_list[event_idx][0]
        event_boxes = bbx_list[event_idx][0]
        event_gt = gt_list[event_idx][0]
        for image_idx in range(len(event_files)):
            filename = str(event_files[image_idx][0][0])
            all_boxes = event_boxes[image_idx][0]
            selected_indices = event_gt[image_idx][0]
            # gt_list indices are 1-indexed (MATLAB convention).
            selected = [tuple(int(v) for v in all_boxes[i - 1]) for i in selected_indices.flatten()]
            keep_boxes[f"{event_name}/{filename}.jpg"] = selected
    return keep_boxes


def _test_split_relpaths() -> set[str]:
    """Resolve our test split's images back to their event/filename.jpg relative paths."""
    test_images_dir = RAW_VAL_IMAGES_DIR.parent.parent.parent / "processed" / "images" / "test"
    relpaths = set()
    for image_link in test_images_dir.iterdir():
        target = image_link.resolve()
        # .as_posix() (always "/"), not str() -- str() renders "\" on Windows,
        # which would never match _load_keep_boxes' "event/filename.jpg" keys.
        relpaths.add(target.relative_to(RAW_VAL_IMAGES_DIR).as_posix())
    return relpaths


def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Intersection-over-union between two (x, y, w, h) boxes."""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _match_recall(gt_boxes: list[tuple[int, int, int, int]], pred_boxes: list[tuple[float, float, float, float]]) -> int:
    """Count how many gt_boxes have at least one unmatched prediction with IoU >= threshold (greedy)."""
    used_preds: set[int] = set()
    matched = 0
    for gt in gt_boxes:
        best_iou, best_pi = 0.0, -1
        for pi, pred in enumerate(pred_boxes):
            if pi in used_preds:
                continue
            iou = _iou(gt, pred)
            if iou > best_iou:
                best_iou, best_pi = iou, pi
        if best_iou >= IOU_THRESHOLD:
            matched += 1
            used_preds.add(best_pi)
    return matched


def official_difficulty_recall(model: YOLO, device: str) -> dict:
    """Compute recall against the official Easy/Medium/Hard partition, restricted to our test split."""
    test_relpaths = _test_split_relpaths()
    results = {}

    for difficulty in ("easy", "medium", "hard"):
        keep_boxes_by_image = _load_keep_boxes(GROUND_TRUTH_DIR / f"wider_{difficulty}_val.mat")

        total_gt, total_matched, images_evaluated = 0, 0, 0
        for relpath in test_relpaths:
            gt_boxes = keep_boxes_by_image.get(relpath, [])
            if not gt_boxes:
                continue
            images_evaluated += 1

            image_path = RAW_VAL_IMAGES_DIR / relpath
            prediction = model.predict(source=str(image_path), device=device, verbose=False)[0]
            pred_xywh = prediction.boxes.xywh.cpu().tolist()  # center-x, center-y, w, h
            pred_boxes = [(cx - w / 2, cy - h / 2, w, h) for cx, cy, w, h in pred_xywh]

            total_gt += len(gt_boxes)
            total_matched += _match_recall(gt_boxes, pred_boxes)

        results[difficulty] = {
            "images_evaluated": images_evaluated,
            "gt_faces": total_gt,
            "recall": round(total_matched / total_gt, 4) if total_gt else None,
        }
    return results