"""
appearance.py

Turns a person's body crop into a vector that can be compared against
another camera's view of them.

This is what geometry cannot do. Two cameras watching the same room can
recognise one person from position alone, because both report into the
same world frame -- but only while they see them at the same moment.
Someone who leaves one camera's view and enters another's a few seconds
later has no overlapping position to match on, and appearance is the only
remaining evidence.

What it encodes is clothing, build and colouring, not identity in any
durable sense: change your jacket and you are a new person to it. That
bounds what it can do -- continuity within a visit, not recognition of a
returning customer -- and it also bounds the harm, which is why body
appearance was chosen over face embeddings for this.

NOT WIRED INTO THE PIPELINE. The machinery here is sound and the backbone
is not, and the measurement that says so is worth keeping next to the
code rather than in a commit message.

The backbone is a torchvision ResNet18 pretrained on ImageNet, global
average pooled and L2-normalised. It was never trained to tell people
apart, and on real body crops from the evaluation clips it does not:

    same person, same clip      cosine 0.430 - 0.999
    same person, different clip cosine 0.483 - 0.920

Those ranges overlap completely, and every crop involved is one person.
There is no threshold that separates "same track" from "different track",
because what the vector mostly encodes is pose, scale and lighting.
Clustering on it would merge and split people close to at random, which
is worse than the double-counting it was meant to fix -- that at least is
systematic.

Two things have to change before this is usable. The backbone has to be
one trained for person re-identification (OSNet and its relatives), which
drops straight into embed() and is the only part that would change. And
there has to be footage containing several different people to tune the
threshold against: every clip currently in the repository shows one
person, so the separation that matters -- between two people, not between
two views of one -- has never been measured at all.

Kept because the surrounding machinery is the reusable part: per-track
averaging, the crop-size floor, normalisation, retirement. Swapping the
backbone is a one-line change to __init__ once both of the above hold.

Embeddings are averaged over a track rather than taken from one frame.
A single crop catches whatever the person happened to be doing -- turned
away, half-occluded, motion-blurred -- and the mean over a track is both
steadier and cheaper to compare than a set of vectors.
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

# The usual person re-identification crop shape: people are about twice as
# tall as they are wide, and squashing them to a square throws that away.
CROP_SIZE = (256, 128)

# Below this many pixels a crop carries no usable appearance -- it is a
# smear of a few colours, and embedding it would add noise to the track's
# running mean rather than evidence.
MIN_CROP_PIXELS = 32


class AppearanceEmbedder:
    """Produces a comparable appearance vector from a person's body crop."""

    def __init__(self, device: str | None = None) -> None:
        """Load the backbone once, with its classification head removed."""
        self.device = device or "cpu"
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        # Drop the 1000-way classifier: the pooled features before it are the
        # description of the crop, and the class scores are about ImageNet.
        model.fc = torch.nn.Identity()
        self._model = model.eval().to(self.device)
        self._prepare = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(CROP_SIZE, antialias=True),
                transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
            ]
        )

    def embed(self, crop: np.ndarray) -> np.ndarray | None:
        """Return an L2-normalised appearance vector for a BGR body crop, or None if it is unusable.

        Normalised so that comparing two vectors is a dot product, and so
        that averaging several keeps them on the same scale.
        """
        if crop.size == 0 or min(crop.shape[:2]) < MIN_CROP_PIXELS:
            return None
        rgb = crop[:, :, ::-1].copy()
        with torch.no_grad():
            batch = self._prepare(rgb).unsqueeze(0).to(self.device)
            features = self._model(batch).squeeze(0).cpu().numpy()
        norm = np.linalg.norm(features)
        return None if norm == 0 else (features / norm).astype(np.float32)


class TrackAppearance:
    """Keeps one running appearance vector per track, averaged over the frames seen so far."""

    def __init__(self, embedder: AppearanceEmbedder) -> None:
        """Start with no tracks described."""
        self._embedder = embedder
        self._sums: dict[int, np.ndarray] = {}
        self._counts: dict[int, int] = {}

    def observe(self, track_id: int, crop: np.ndarray) -> None:
        """Fold one more view of a track into its running appearance."""
        vector = self._embedder.embed(crop)
        if vector is None:
            return
        if track_id in self._sums:
            self._sums[track_id] += vector
            self._counts[track_id] += 1
        else:
            self._sums[track_id] = vector.copy()
            self._counts[track_id] = 1

    def embedding(self, track_id: int) -> np.ndarray | None:
        """Return the track's mean appearance, re-normalised, or None if it has none yet."""
        total = self._sums.get(track_id)
        if total is None:
            return None
        mean = total / self._counts[track_id]
        norm = np.linalg.norm(mean)
        return None if norm == 0 else (mean / norm).astype(np.float32)

    def observations(self, track_id: int) -> int:
        """How many crops went into this track's appearance, so a caller can weigh how settled it is."""
        return self._counts.get(track_id, 0)

    def retire(self, active: set[int]) -> None:
        """Forget tracks no longer being reported, so memory follows the room rather than the uptime."""
        for track_id in list(self._sums):
            if track_id not in active:
                del self._sums[track_id]
                del self._counts[track_id]
