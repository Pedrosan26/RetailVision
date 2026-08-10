"""
generate_aruco_markers.py

One-off generator for printable ArUco marker images, for physically
marking a zone's boundary corners. Uses DICT_4X4_50 (50 unique IDs, a
simple/small pattern that stays reliably detectable at a greater
distance than a busier dictionary) -- plenty of headroom for a handful
of zones at 4 markers each. Not part of the live pipeline; run once per
new zone you want to physically mark.

Each PNG carries a white quiet zone around the black pattern, because the
detector finds a marker by its black border against a light surround and
will simply not see one that is cropped flush or laid on a dark surface.
Do not trim that white margin off when cutting the printouts.

Pose estimation measures against the black square only, so --marker-size
elsewhere is the black square's side, not the paper's.

Usage: PYTHONPATH=. ./venv/bin/python3 scripts/generate_aruco_markers.py --ids 0 1 2 3 --out markers/
Print every marker at the same physical size and place one at each corner
of the zone's floor area. Larger is markedly better for pose accuracy:
measured on synthetic views, a 10cm marker is unreliable past about 4m,
while 20cm holds to a few centimeters across a normal room.
"""

import argparse
from pathlib import Path

import cv2

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
MARKER_PIXELS = 600  # high-res output so printing at any reasonable physical size stays sharp
QUIET_ZONE_RATIO = 0.25  # white margin per side, as a fraction of the marker's side


def parse_args() -> argparse.Namespace:
    """Parse --ids (marker IDs to generate) and --out (output directory)."""
    parser = argparse.ArgumentParser(description="Generate printable ArUco marker PNGs")
    parser.add_argument("--ids", type=int, nargs="+", required=True, help="Marker IDs to generate (0-49)")
    parser.add_argument("--out", type=Path, default=Path("markers"), help="Output directory (default: markers/)")
    return parser.parse_args()


def main() -> None:
    """Generate one PNG per requested marker ID, each framed by a white quiet zone."""
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    margin = int(MARKER_PIXELS * QUIET_ZONE_RATIO)
    for marker_id in args.ids:
        image = cv2.aruco.generateImageMarker(DICTIONARY, marker_id, MARKER_PIXELS)
        framed = cv2.copyMakeBorder(image, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)
        out_path = args.out / f"marker_{marker_id}.png"
        cv2.imwrite(str(out_path), framed)
        print(f"Wrote {out_path}")
    print(f"Black square is {1 / (1 + 2 * QUIET_ZONE_RATIO):.0%} of the image side -- measure that, not the paper.")


if __name__ == "__main__":
    main()
