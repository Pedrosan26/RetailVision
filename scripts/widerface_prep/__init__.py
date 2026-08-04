"""
widerface_prep

Package that turns the raw WIDER FACE dataset into a YOLOv8 detection
dataset for face-detector training. Unlike utkface_prep/fer2013_prep
(which lay out pre-cropped single-face images into classification class
folders), this package produces bounding-box annotations, since face
*localization* is the thing being trained here. See
scripts/prepare_widerface.py for the CLI entry point that orchestrates
the submodules in this package.
"""