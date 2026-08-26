"""
generate_chessboard.py

One-off generator for the printable chessboard that camera calibration
needs (see scripts/setup/calibrate_camera.py). Sized by inner-corner count, the
same way OpenCV and calibrate_camera.py's --pattern argument describe a
board: a 9x6 board has 9 by 6 interior corners, so 10 by 7 squares.

The board is framed by a white margin, because corner detection needs the
outer squares to sit against a light surround rather than run off the edge
of the paper.

Print it and mount it on something rigid -- cardboard, clipboard, foam
board. A sheet held in the hand bows slightly, and since calibration is
solving for lens distortion, a bowed board is indistinguishable from a
distorted lens and quietly corrupts the result. Flatness matters much more
than the printed square size, which does not affect the lens model at all.

Usage: PYTHONPATH=. ./venv/bin/python3 scripts/setup/generate_chessboard.py --out calibration/chessboard.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

DEFAULT_PATTERN = "9x6"
SQUARE_PIXELS = 180  # high-res output so the printed edges stay crisp
MARGIN_SQUARES = 0.6  # white surround, in squares, so the outer corners are detectable


def parse_pattern(text: str) -> tuple[int, int]:
    """Parse a 'COLUMNSxROWS' inner-corner count, e.g. '9x6' for a 10x7 square board."""
    columns, _, rows = text.partition("x")
    return (int(columns), int(rows))


def parse_args() -> argparse.Namespace:
    """Parse the board's inner-corner pattern and where to write the printable image."""
    parser = argparse.ArgumentParser(description="Generate a printable chessboard for camera calibration")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help=f"Inner corners as COLUMNSxROWS (default: {DEFAULT_PATTERN})")
    parser.add_argument("--out", type=Path, default=Path("calibration/chessboard.png"), help="Where to write the PNG")
    return parser.parse_args()


def build_board(pattern_size: tuple[int, int]) -> np.ndarray:
    """Render a chessboard with the given inner-corner count, framed by a white margin."""
    columns, rows = (n + 1 for n in pattern_size)
    squares = (np.indices((rows, columns)).sum(axis=0) % 2).astype(np.uint8)
    board = np.kron(squares, np.ones((SQUARE_PIXELS, SQUARE_PIXELS), dtype=np.uint8)) * 255
    board = board.astype(np.uint8)
    margin = int(SQUARE_PIXELS * MARGIN_SQUARES)
    return cv2.copyMakeBorder(board, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)


def main() -> None:
    """Generate the printable chessboard and confirm OpenCV can find its corners in it."""
    args = parse_args()
    pattern = parse_pattern(args.pattern)
    board = build_board(pattern)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), board)
    print(f"Wrote {args.out} ({pattern[0]}x{pattern[1]} inner corners, {pattern[0]+1}x{pattern[1]+1} squares)")

    found, _ = cv2.findChessboardCorners(board, pattern)
    print(f"Corner detection on the generated file: {'ok' if found else 'FAILED'}")
    print("Print it, then mount it flat on something rigid before calibrating.")


if __name__ == "__main__":
    main()
