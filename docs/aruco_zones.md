# Marker-based zones

How a retail floor gets divided into zones that several cameras can agree
on, using printed ArUco markers.

## The problem this solves

The original people counter uses a virtual line: a track that crosses it
counts as an entry, and the net total is occupancy. Two things break that
at store scale. Occupancy drifts, because a single missed crossing skews
the running total permanently with nothing to correct it. And a line
describes a doorway, not an area, so it cannot answer "how many people are
in the electronics section right now."

The obvious replacement is a polygon per zone and a point-in-polygon test,
which reports a live headcount instead of accumulating crossings. The hard
part is not the polygon, it is agreeing on where the polygon *is* when
more than one camera is watching, and when no single camera can see the
whole area.

## Why not a homography

The first attempt mapped image pixels straight onto the zone's plane with
a homography. It works, and needs no camera calibration at all, but a
homography is only solvable from four point correspondences at once. That
means one camera has to see all four of a zone's corner markers
simultaneously. In a real room with shelving, that view frequently does
not exist. Worse, two cameras could only be related to each other by
sharing all four markers, which defeats the point of using two.

## What replaced it

Pose estimation solves each marker independently. A marker's four corners,
plus its known printed size, are already a complete 3D-to-2D
correspondence set on their own, so `cv2.solvePnP` recovers that marker's
full position and orientation relative to the camera from that one marker.

The printed size is the only measurement in the entire system. The room is
never measured, distances between markers are never measured, and nothing
has to be re-measured when a zone is moved or resized -- which was the
explicit requirement, since the zone perimeter is not constant across
deployments.

From there:

- One marker is fixed as the **anchor**, defining the world origin. With
  markers laid flat on the floor, world z=0 is the floor and world x/y are
  floor coordinates, in meters.
- A camera that can see any one already-mapped marker can compute where it
  itself is standing in that world frame.
- From that position, every other marker it can see gets added to the map.

So the map grows outward from the anchor, one overlapping view at a time.
**Two cameras need one marker in common, not four.** A zone larger than
any single camera's view still ends up fully described, with different
corners contributed by different cameras.

## The ambiguity that has to be handled

A flat square viewed through a camera is genuinely ambiguous. Two
different orientations, roughly mirrored through the marker's own plane,
project to almost the same four pixels. `solvePnP` returns both, and their
reprojection errors can be nearly identical:

```
marker 1, 20cm print, 5m away:
  candidate 0: reprojection 0.2240px   orientation error 128.4 degrees
  candidate 1: reprojection 0.2297px   orientation error   0.8 degrees
```

Taking the lower reprojection error picks the 128-degree-wrong one, by a
margin of 0.006px. Position survives this -- both candidates put the
marker in about the same place -- but orientation does not, and because
the map chains marker to camera to marker through full orientations, a
single wrong pick corrupts everything mapped afterwards. In synthetic
tests this turned a 2x3m zone into a 9-metre error.

`CameraLocalizer` resolves it with three signals, in order:

1. **Other markers in view.** A flipped orientation still explains its own
   marker's corners, but predicts the other markers in view badly. Scoring
   a candidate camera pose against every mapped marker separates them
   decisively.
2. **The mounting plane.** While only the anchor is mapped, signal 1
   cannot help, since both candidates explain that one marker equally. But
   the markers all being on a common plane does separate them: the wrong
   orientation lifts every *other* marker in view off that plane and tilts
   it, which is measurable directly.
3. **Continuity.** If both survive, prefer whichever is closest to where
   this camera was last frame. A fixed camera does not jump.

With all three, orientation came out correct in every synthetic
configuration tested, and the residual error is ordinary measurement
noise.

## Accuracy, and how big to print

Measured on synthetic renders through a 900px-focal camera. "Lateral" is
error across the camera's view, "depth" is error along it -- depth is
always the weaker axis for a planar target, which is inherent to the
method rather than a defect.

| Printed size | Useful range | Typical depth error | Typical lateral error |
| --- | --- | --- | --- |
| 10cm | to ~4m | 1-3cm | ~1mm |
| 20cm | across a normal room | ~1cm | ~1mm |

End-to-end, reconstructing a 2x3m zone from two cameras sharing exactly
one marker: 20cm markers placed all four corners within about 6cm, and
every point-in-zone test came out correct. 10cm markers were not reliably
detected at all by the camera 5m from the shared marker.

**Print at 20cm if the room allows it.** Marker size matters far more than
how many markers the cameras share.

Two detection details that are easy to get wrong:

- **Keep the white margin.** The detector finds a marker by its black
  border against a light surround. A marker cropped flush to the black
  edge, or laid on a dark surface, is simply not detected.
  `generate_aruco_markers.py` includes the margin -- do not trim it off.
- **`--marker-size` is the black square's side**, not the paper's.

Detection itself also runs with sub-pixel corner refinement enabled, which
the detector does not do by default. Whole-pixel corners are ample for
reading a marker's ID but not for solving its pose; enabling refinement
roughly halved pose error in testing.

## Setting it up

1. **Generate and print markers.** One ID per zone corner, all at the same
   physical size, IDs unique across all zones.

   ```
   PYTHONPATH=. ./venv/bin/python3 scripts/generate_aruco_markers.py --ids 0 1 2 3 --out markers/
   ```

2. **Calibrate each camera, once.** This describes the camera itself, not
   the room, so it stays valid across every zone and deployment that
   camera is used in. Rerun only if the lens, resolution or zoom changes.

   ```
   PYTHONPATH=. ./venv/bin/python3 scripts/calibrate_camera.py --source 0 --out calibration/camera_0.json
   ```

   Hold a printed chessboard at clearly different distances, tilts and
   positions, including near the frame corners where distortion is
   strongest. Views that are all head-on and centered constrain the model
   poorly however many are captured. A reprojection error below about
   0.5px is a good result.

3. **Lay the markers flat at the zone's floor corners**, and define the
   zone by their IDs. Copy `config/zones.example.json` to
   `config/zones.json` and edit.

4. **Check it live.**

   ```
   PYTHONPATH=. ./venv/bin/python3 scripts/aruco_pose_test.py \
       --source 0 --calibration calibration/camera_0.json \
       --marker-size 0.20 --zones config/zones.json
   ```

   Pass several sources to test the multi-camera case in one process, with
   one shared map and no networking involved:

   ```
   --source 0 1 --calibration calibration/camera_0.json calibration/camera_1.json
   ```

   Each ready zone's floor polygon is projected back into every camera
   view, which is the visual check that the zone landed where the markers
   physically are. Clicking anywhere back-projects that pixel onto a plane
   at head height and reports the world position and containing zone.

## Why head height, not the floor

This pipeline detects faces, not whole bodies, so the bottom of a
detection box is a chin, not a pair of feet. Intersecting a face's viewing
ray with the floor would place that person well beyond where they actually
stand. Positions are instead back-projected onto a horizontal plane at
roughly head height (`DEFAULT_HEAD_HEIGHT_METERS`), whose x/y still reads
as that person's floor footprint, which is what the zone test needs.

## Module layout

- `calibration.py` -- `CameraCalibration`, the per-camera lens model, and
  the chessboard solver that produces it.
- `marker_pose.py` -- `MarkerPoseEstimator`, detects markers and solves
  each one's pose relative to the camera. `solve_candidates()` returns
  both ambiguous orientations rather than guessing between them.
- `marker_map.py` -- `MarkerMap`, every marker's pose in the shared world
  frame; `CameraLocalizer`, which places a camera in it, resolves the
  ambiguity and extends the map; `project_to_plane()`, pixel to world
  position.
- `zones.py` -- `Zone`/`ZoneMap`, zone polygons assembled from mapped
  marker positions, and point-in-zone occupancy.

## Current limitations

- **Marker poses are first-observation-wins.** A marker's world pose is
  fixed when first seen from a known vantage point and never refined.
  Averaging repeated observations, or a full pose-graph optimization,
  would reduce drift across a long chain of markers -- worth doing once
  that drift is actually measured to be a problem.
- **Zones are assumed convex.** Corners are ordered by angle around their
  centroid so that markers can be listed in any order, which trades a
  shape restriction that does not bite for retail floor areas against a
  setup step that is easy to get wrong.
- **Head height is a constant, not per-person.** A tall and a short person
  are placed slightly differently for the same true floor position. Small
  relative to zone dimensions, but it is a real source of error near a
  boundary.
- **Not yet wired into the live pipeline.** This is spike work: the
  modules and the test script exist, but `pipeline_demo.py` still uses the
  virtual-line counter, and nothing yet populates `zone_id` in the logged
  records. Re-identifying the same person across zones or across cameras
  remains future work, unchanged from `docs/people_counter.md`.
