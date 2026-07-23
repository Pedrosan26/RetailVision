"""
Package that turns the raw UTKFace dataset into a manifest-based regression
dataset (continuous age labels), reusing utkface_prep's filename parsing
and stratified split, but writing a CSV manifest instead of a
folder-per-class layout since regression has no discrete classes.
"""