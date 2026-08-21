# Camera setup

Getting several cameras onto one machine, calibrated, and ready for
marker-based zones. Everything here was measured while bringing up a
three-camera office deployment; the numbers are from that hardware, but
the failure modes are general.

The recurring theme is that **none of these failures announce themselves**.
Cameras open and return nothing, calibrations report good errors while
being unusable, resolution changes look like dead hardware. Each section
below says what the symptom looks like as well as what causes it.

## How many cameras fit on one machine

USB webcams reserve *isochronous* bandwidth when they open, and the
allocation is refused outright rather than shared. So the failure is not
"everything gets slower" -- it is one camera opening successfully and then
delivering no frames at all, while its neighbours are fine.

Measured on a MacBook Air with a 4-port hub:

| Cameras on the hub | Result |
| --- | --- |
| 3 | all healthy, 25-30 fps each |
| 4 | the 4th opens and delivers zero frames |

Two things that catch people out:

- **A USB 3.0 hub contains a separate USB 2.0 hub inside it.** Webcams are
  almost always USB 2.0 devices, so they all share one 480 Mbps upstream
  link no matter how fast the hub is rated. A bigger or powered hub does
  not help; power is not the constraint.
- **A laptop's built-in camera is not on that bus.** Three USB cameras plus
  the built-in works fine and looks like "four cameras", which is
  misleading when planning.

Rough uncompressed costs, for sizing: 640x480@30 is about 147 Mbps,
1280x720@30 about 442 Mbps. One 720p camera nearly saturates USB 2.0 on
its own, which is why lowering resolution moves the limit far less than
expected -- three still fit, four still do not.

**The fix that works is splitting across separate ports**, giving two
independent upstream links. On macOS, `CAP_PROP_FOURCC` is ignored, so the
MJPEG trick that helps on Linux is not available.

Use `scripts/camera_check.py` to test this. It opens every camera at once,
reads each on its own thread, and reports per-camera capture rate,
sharpness and dropped reads.

> Reading several cameras in turn on a single thread makes the measurement
> meaningless: every read waits for the one before it, so one dead camera
> drags the whole set down to its timeout rate. That is what a first
> attempt showed -- four cameras all reporting exactly 0.9 fps and 73
> frames, which looked like bus saturation and was actually one broken
> device blocking each iteration for ~1.1s.

## Resolution

`scripts/probe_cameras.py` reports which resolutions each camera honours
exactly and which it silently substitutes.

Two macOS behaviours worth knowing:

- **Requesting a resolution restarts the capture session**, and the camera
  can take **up to ~9 seconds** to deliver its first frame afterwards.
  Reading immediately looks exactly like a dead or busy camera. Both
  `calibrate_camera.py` and `aruco_pose_test.py` wait this out.
- **Some cameras only work at their native resolution.** A MacBook's
  built-in camera delivered 1920x1080 fine and returned *no frames at all*
  at 1280x720 or 640x480, even after 25 seconds. USB cameras were happy at
  every size.

Cameras do not need to match each other -- each carries its own
calibration. What must match is **each camera's calibration resolution
against the resolution it later runs at**, because focal length is measured
in pixels. `aruco_pose_test.py` and `pipeline_demo.py` read the resolution
out of the calibration file and request it automatically.

## Calibration

One-off per physical camera, via `scripts/calibrate_camera.py`. It
describes the camera and lens, not the room, so it stays valid across
deployments. Redo it only if the resolution, zoom, lens or camera changes.

Print the board with `scripts/generate_chessboard.py` and **mount it flat on
something rigid**. A hand-held sheet bows, and since calibration is solving
for lens distortion, a bowed board is indistinguishable from a distorted
lens.

### What to capture

16 views, varied. SPACE captures, `c` solves once at the end.

| # | Where in frame | How to hold it |
| --- | --- | --- |
| 1-3 | centre | flat-on, at three distances |
| 4-7 | centre | tilted ~35 degrees: top, bottom, left, right edge away |
| 8-11 | one per frame corner | tilted ~30 degrees toward the centre |
| 12-13 | centre | rolled 45 degrees, near and far |
| 14-16 | anywhere | steeper tilts, ~50 degrees |

**Tilt determines focal length.** A board held flat and square-on is
degenerate: a big board far away and a small one close look identical, and
only foreshortening breaks the tie. All-frontal captures produce a
plausible-looking result with a badly wrong focal length.

**Frame corners determine distortion**, which is near zero at the centre
and strongest at the periphery.

### Working distance

The board wants to fill roughly 40% of the frame width, leaving room to
push it into the corners and tilt it steeply without clipping. That
distance depends on the lens:

| Lens | Board fills 40% at (20cm board) |
| --- | --- |
| 89 degrees | 0.25m |
| 67 degrees | 0.38m |
| 62 degrees | 0.41m |
| 50 degrees | 0.54m |

A narrow lens magnifies more, so the board overflows the frame sooner and
corner shots become impossible -- which is exactly the capture pattern that
leaves `fx` and `fy` under-constrained. If a camera shows an `fx`/`fy`
split above about 1%, work further back.

### Judging the result

**Reprojection error alone does not establish that a calibration is
usable.** It only says the model fits the views it was solved from, which
is what an overfitted model does best.

Three of four cameras in this deployment reported errors between 0.31 and
0.86px while being unusable. Their radial coefficients had absorbed capture
noise as enormous values that cancel within the captured views and diverge
towards the frame corners:

| Camera | reprojection | k3 | round-trip error |
| --- | --- | --- | --- |
| 0 | 0.475px | -12.65 | 14,412px |
| 1 | 0.355px | -12.24 | 110px |
| 2 | 0.446px | +165.66 | 302px |
| 3 | 0.858px | -0.645 | 0.00px |

Solving a pose undistorts the image points; reprojecting distorts them
again. When the coefficients are physically absurd those two operations
disagree, so a pose fits its own corners perfectly and lands hundreds of
pixels away when projected back. Downstream this looks like a broken marker
map for no visible reason.

`calibrate_camera.py` now solves richest-first and keeps the first
distortion model that is self-consistent, dropping k3, then k2, then the
tangential terms. Check both numbers it prints:

- **Reprojection error** under 0.5px is good, over 1.0px is not usable.
- **Lens-model self-consistency** must say `ok`. If it says BROKEN, recapture.

Two other signals worth a glance:

- **`fx` vs `fy`** should agree within ~1%; sensors have square pixels. A
  persistent split that survives a recapture may be real (non-uniform
  scaling in the camera's output pipeline), but a first occurrence usually
  means poor frame coverage.
- **Optical centre** should sit within a few percent of the image centre. A
  large offset means the board never reached the frame edges.

After recalibrating the three broken cameras, live marker-map fits fell
from 122 and 378px to **0.36px and 0.48px** with no physical change to the
setup.

## Throughput with several cameras

Each camera node is its own process running the full pipeline -- face
detection plus the age/gender and emotion classifiers -- so three cameras
means three copies of all of it.

Measured running three concurrently on a MacBook Air (MPS), ~50 seconds
each:

| Faces per frame | FPS |
| --- | --- |
| 0.10 | 14.1 |
| 1.18 | 10.5 |
| 1.85 | 9.3 |

About 34 FPS of aggregate inference across the three streams.

**Throughput falls with face count, not just camera count.** The detector
runs once per frame regardless, but the classifiers run once per detected
face, so a busy view costs materially more than an empty one. Benchmark
with people actually present; an empty room flatters the numbers by ~50%.

Capture is not the limit here -- `camera_check.py` showed the same cameras
sustaining 25-30 fps each without inference. The gap is compute. Run
`camera_check.py --detect` to see both effects together and tell them
apart: if cameras are healthy without it and starve with it, the machine is
compute-bound rather than bus-bound, and those have different fixes.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| Camera opens, delivers no frames, others fine | bus bandwidth -- too many on one hub |
| Every camera reports the same low fps | a dead camera stalling a serial read loop |
| "0 views captured" right after starting calibration | another process holds the camera, or a resolution change is still restarting the session |
| Calibration looks fine, marker map is nonsense | distortion model not self-consistent |
| Distances all scaled by a constant factor | `--marker-size` does not match the printed black square |
| Marker detected but pose jumps around | too small or too oblique; see `docs/aruco_zones.md` |
