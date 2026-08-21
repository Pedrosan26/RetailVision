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

The printed size is one of only two measurements in the entire system; the
other is how high the anchor marker sits above the floor. The room is never
measured, distances between markers are never measured, and nothing has to
be re-measured when a zone is moved or resized -- which was the explicit
requirement, since the zone perimeter is not constant across deployments.

From there:

- One marker is fixed as the **anchor**, defining the world origin. World
  z=0 is placed on the floor, `--marker-height` below the anchor, so world
  x/y are floor coordinates in meters and z is height above the floor.
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
2. **The mounting.** While only the anchor is mapped, signal 1 cannot
   help, since both candidates explain that one marker equally. How the
   markers are physically mounted does separate them: the wrong orientation
   points a marker's up-axis at the ground instead of the ceiling. For
   floor markers that axis is the normal, and their shared plane gives an
   extra term; for wall markers it is the printed up direction, which holds
   even across walls facing different ways. See "Where the markers are
   mounted".
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

## Where the markers are mounted

Markers can lie flat on the floor or stand upright on walls, and the two
need different handling. Pass `--marker-mounting floor` or `wall`.

**Wall mounting is usually the practical choice.** Cameras sit at roughly
human height, so a floor marker is seen at a steep grazing angle -- often
past 80 degrees off face-on, where its image degenerates towards a line and
corner localization becomes hopeless. Upright markers face the cameras.

Two consequences follow from the mounting:

- **Which axis points up.** Resolving a square's two-fold pose ambiguity
  needs to know which orientation is physically possible. Flat markers
  point their *normal* at the ceiling; upright markers point their printed
  *up direction* there. Markers on four different walls are neither
  coplanar nor co-facing, so the coplanarity assumption fails outright for
  them -- but they are all still upright, and that is enough.
- **Where the floor is.** The world frame is anchored on one marker, so a
  marker partway up a wall would put z=0 at its own height: zone polygons
  float in mid-air at marker level, and back-projecting a person onto a
  head-height plane aims above the camera instead of below it.
  `--marker-height`, the anchor's measured height above the floor, moves
  z=0 down to the real floor.

**A zone's corners must enclose the area you want to measure.** Markers
along a single wall all lie in one vertical plane, so projected down to the
floor they collapse onto a line and bound nothing. One marker per wall
around a room gives a proper polygon; several on one wall do not.

## Setting it up

Camera-side practicalities -- how many cameras fit on one machine,
calibration technique, and how to tell a usable calibration from one that
merely reports a good number -- are in `docs/camera_setup.md`.

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
   poorly however many are captured.

   **Check the self-consistency line, not just the reprojection error.**
   Reprojection error only says the model fits the views it was solved
   from, which is exactly what an overfitted model does best. Three of the
   four cameras this was developed against produced errors between 0.31 and
   0.86px while being unusable: their radial coefficients had absorbed
   noise as enormous values (k3 of -12.6, -12.2, +165.7) that cancel within
   the captured views and diverge towards the frame corners. Solving a pose
   undistorts and reprojecting distorts again, and such a model makes those
   two disagree by hundreds of pixels -- which shows up downstream as a
   marker map that looks broken for no visible reason.

   The script reports both, simplifies the distortion model until it is
   self-consistent, and refuses to endorse one that is not. Aim for
   reprojection under 0.5px **and** self-consistency `ok`.

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
- **Zones are convex by construction.** The polygon is the convex hull of
  the markers' floor positions -- the fence around them all -- so neither
  marker IDs nor listing order have to match physical adjacency, and a
  marker whose floor position lands inside the shape the others form
  (survey noise, or a marker on an interior pillar) cannot fold a dent
  into the perimeter. The cost is that a deliberately concave zone cannot
  be expressed; retail floor areas are effectively rectangles, so that
  restriction does not bite.
- **Marker poses are not reconciled when chains meet.** Two cameras can
  reach the same marker by different routes; nothing detects or corrects a
  disagreement between them. A loop closure or bundle adjustment would,
  and is the natural next step if accuracy stops being adequate.
- **Head height is a constant, not per-person.** A tall and a short person
  are placed slightly differently for the same true floor position. Small
  relative to zone dimensions, but it is a real source of error near a
  boundary.
- **Re-identification across cameras is still future work.** Two cameras
  watching one zone each count the people they can see; nobody notices that
  a person visible to both is one person, so their counts are not additive.
  Unchanged from `docs/people_counter.md`.


## Placing cameras and markers

Three separate geometric constraints decide whether the numbers coming out
are trustworthy. All were measured on the deployment this was built
against.

### How big a marker looks

A marker's apparent side in pixels is `fx * marker_size / distance`. Below
roughly **50px a side** there are too few pixels to localize corners
precisely, and beyond about **60 degrees off face-on** the square
degenerates towards a line and its corners stop being well separated.
`aruco_pose_test.py` flags both.

Detection keeps working well past those thresholds -- markers are still
recognised at 22px and 84 degrees. What degrades is the *pose*, and only
the pose. That distinction matters: a marker being detected is not evidence
that its position is good.

With a 14cm marker on a 1500px-focal camera:

| Distance | Apparent side | Verdict |
| --- | --- | --- |
| 1m | 210px | excellent |
| 3m | 70px | good |
| 5m | 42px | marginal |
| 8m | 26px | unusable for pose |

Range scales linearly with printed size, so 28cm markers double every
figure above. **Marker size is the cheapest lever available** -- far more
effective than adding more markers.

### What a small marker costs

Error at a distant marker is amplified by the lever arm out to everything
mapped through it. Measured on a 32px marker at 7.08m:

```
1px of corner error   ->  1.79 degrees of camera orientation error
over 2.89m to the next marker  ->  9.0cm of position error
which reprojects as   ->  49px
```

So an observed 24px inconsistency across two mapped markers corresponds to
**well under one pixel** of corner noise on the far one. Nothing is
malfunctioning at that point; the marker is simply too small to anchor
anything.

### Camera height, for markers on the floor

A camera at height `h` and horizontal distance `d` from a floor marker sees
it `90 - atan(h/d)` degrees off face-on. Cameras at human height therefore
see floor markers at grazing angles -- 84 degrees was measured for a camera
0.3m above a marker 3m away, which is unusable.

Maximum horizontal distance that still passes both thresholds, 14cm markers:

| Camera height | Max distance to marker |
| --- | --- |
| 1.0m | 1.7m |
| 1.5m | 2.0m |
| 2.0m | 2.1m |
| 3.0m | 1.9m |

Height stops helping past ~2.5m, because the camera then recedes in
straight-line distance faster than it gains angle. **Upright wall markers
avoid this constraint entirely**, which is usually the better answer.

### Camera height, for locating people

This is the constraint that most affects zone accuracy, and the least
obvious.

A person's position is where the ray through their detected face crosses a
horizontal plane at `--head-height`. If their face is not at that height,
the error is:

```
reported distance = true distance x (h_camera - h_assumed) / (h_camera - h_face)
```

The closer the camera sits to the assumed plane, the more violently that
diverges. Camera at 1.9m, person truly 3.0m away:

| Their face at | plane 1.6m (30cm below camera) | plane 1.2m (70cm below camera) |
| --- | --- | --- |
| assumed -10cm | 2.25m (-0.75m) | 2.62m (-0.38m) |
| assumed | 3.00m | 3.00m |
| assumed +10cm | 4.50m (+1.50m) | 3.23m (+0.23m) |

A person 10cm taller than assumed is reported **1.5m further away** with a
30cm gap, and 0.23m further with a 70cm gap. Someone seated where standing
was assumed lands metres out.

Two consequences:

- **Set `--head-height` to the posture you are actually measuring.** An
  office of seated people wants ~1.2m, not the 1.6m standing default. On
  the setup above that alone cut worst-case error from 1.50m to 0.50m.
  This affects the reported *coordinate* only: zone *membership* no longer
  rides on it. The face lies somewhere on the viewing ray, and where
  depends on posture, so membership samples that ray across the whole
  plausible band of face heights (0.95-1.85m) and accepts any sample
  landing in a zone -- a standing person whose single-plane projection
  overshoots the polygon still counts. A coordinate has to commit to one
  plane; membership does not.
- **Raise the cameras.** At 3.0m the same 10cm head-height error costs
  0.23m instead of 1.50m. Ceiling mounting is standard for this reason.

Detection noise matters far less: a 1px shift in the face box moves the
reported position 2-5cm at a 30cm gap, and 1-2cm at 90cm. **Posture
assumptions dominate, not pixels.**

`pipeline_demo.py` prints each detection's world position beside its box
and warns when a camera is under 0.75m above the head-height plane.

### Bridging cameras

Two cameras are joined by a marker they can both see *at the same moment*.
That bridge marker's pose quality sets the accuracy of everything the
second camera contributes, so **it should be the best-observed marker
available to both**, not whichever happens to be shared.

A bridge that is far from one camera and steeply angled to the other is the
worst case, and produces symptoms that look like a broken map: impossible
camera positions, reprojection errors in the hundreds. Check the per-marker
lines in `aruco_pose_test.py` before concluding anything else is wrong.

Marker IDs must be unique -- one ID means one physical marker. Two
printouts of the same ID in different places are fused into a single wrong
position, silently.

## Running the live pipeline with zones

Zones are opt-in. Without `--zones` the pipeline behaves exactly as it did
before, using the virtual-line counter and logging `zone_id` as null.

```
PYTHONPATH=. ./venv/bin/python3 -m src.retailvision.pipeline_demo \
    --source 0 \
    --zones config/zones.json \
    --calibration calibration/camera_0.json \
    --marker-size 0.14 --anchor 3 \
    --marker-mounting wall --marker-height 1.70
```

`--calibration` is required alongside `--zones`, because zone positions are
measured in real units and there is no way to get those from pixels alone.

What changes in the output:

- **`zone_id`** is the zone that detection was standing in, or null if the
  camera could not see a mapped marker at that moment. Null therefore means
  "not known", which is deliberately distinct from being outside every
  zone.
- **`count`** becomes that zone's live headcount rather than the virtual
  line's running total. This is the point of the whole exercise: a
  headcount is recomputed every frame and cannot drift, whereas a net count
  is permanently wrong after a single missed crossing.
- The preview labels each detection with its zone and lists the per-zone
  counts, replacing the counting line overlay.

Each camera node runs its own process with its own calibration. Nodes
sharing a zone all report against the same `zone_id`, which is what lets
the server group them -- though see the limitation above about their counts
not being additive while re-identification is missing.
