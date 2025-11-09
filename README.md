# SIF408 Camera Inspection System

This repository implements a modular camera inspection system for SIF408 that:

- Captures images via a Raspberry Pi camera (or static test images).
- Runs a tuned computer-vision algorithm to determine which containers need re-correction.
- Exposes the results to a UR robot (or PLC) over Modbus TCP.
- Provides tools for dataset collection, offline tuning, simulation, and visualisation.

The design goal is:

> All computer-vision logic and parameters live in **one place** (`lid_detection.py`), and all other scripts consume that API without duplicating thresholds or image-processing code.

---

## Repository Overview

### `lid_detection.py`

Core image-processing module. This is the single source of truth for:

- Region-of-interest definitions
- Thresholds
- Edge and line detection settings
- Decision rules for “needs correction”

**Public API**

- `detect_containers(image_bgr) -> (left_needs_correction, right_needs_correction)`

  - Input: Full BGR image (as from `cv2.imread` or a camera capture).
  - Output: Tuple of booleans:
    - `True` = container needs correction.
    - Index 0 = left ROI, index 1 = right ROI.

- `detect_containers_detailed(image_bgr) -> (left_result, right_result)`

  Each result is an `ROIResult`:

  - `box`: `(x1, y1, x2, y2)` ROI in image coordinates.
  - `status`: `LidStatus` with:
    - `needs_correction`
    - `has_lid_like_content`
    - `has_top_line`
    - `mean_angle` (deg)
    - `angle_std` (deg)
    - `reason` (`no_white_content`, `no_lines_detected`, `insufficient_horizontal_lines`, `curved_or_noisy`, `tilted`, `level`)
  - `lines`: list of line segments (ROI-local).
  - `angles`: list of corresponding angles (deg).

**Algorithm (per side)**

1. Crop fixed fractional ROI:
   - `ROI_TOP_FRAC`, `ROI_BOTTOM_FRAC`
   - `LEFT_ROI_*`, `RIGHT_ROI_*`
2. Convert to grayscale and check white pixel fraction:
   - If below `WHITE_MIN_FRACTION`, treat as “no container / no lid” → no correction.
3. Median blur.
4. Canny edge detection.
5. Probabilistic HoughLinesP for near-horizontal segments.
6. If not enough segments: assume OK.
7. Otherwise:
   - If angle spread too large → `curved_or_noisy` → needs correction.
   - Else if mean angle exceeds tolerance → `tilted` → needs correction.
   - Else → `level` → no correction.

Any script needing lid decisions should import from here rather than reimplement logic.

Example:

```python
from lid_detection import detect_containers
import cv2

img = cv2.imread("some_view.jpg")
left_bad, right_bad = detect_containers(img)
```

---

### `tuning_test.py`

Offline tuning and visualisation harness.

Use this to:

- Validate changes to `lid_detection.py`
- Visualise what the algorithm is doing
- Track accuracy over time as parameters are tuned

**Behaviour**

- Reads labelled images from `./tuning/`.
- Expected labels are encoded in filenames:
  - 1st char: left container → `T` (needs correction) or `F` (OK)
  - 2nd char: right container → `T` or `F`
  - Any extra characters are ignored.
  - Examples:
    - `TFm.jpg` → `(True, False)`
    - `FF_01.jpg` → `(False, False)`
- For each valid image:
  - Calls `detect_containers_detailed` from `lid_detection.py`.
  - Compares predictions to expected labels.
  - Prints `[OK]` / `[FAIL]` per image.

**Visualisation**

- Saves annotated copies into `./visualised/`:
  - ROI rectangles (green/red depending on need).
  - Lines used for angle estimation.
  - Text overlays: predicted vs expected, angles, reasons.

**History Logging**

- Appends each run to `tuning_history.csv`:
  - `run_index`, timestamp
  - Total images, correct predictions, accuracy
  - Snapshot of key parameters imported from `lid_detection.py`
- Prints a summary of the last N runs so you can see if changes improved or worsened performance.

**Run**

```bash
python3 tuning_test.py
```

---

### `save_photos_for_calibration.py`

Capture helper used to build the `tuning/` dataset on the Pi.

**Behaviour**

- Uses `Picamera2` (if available) to capture stills interactively.
- Saves images into `SAVE_DIR` (typically `tuning/`) with timestamped filenames.
- Simple key controls:
  - Space: capture and save a frame.
  - `q` / `ESC` / `Ctrl+C`: exit.

**Usage**

```bash
python3 save_photos_for_calibration.py
```

After capture, manually rename images in `tuning/` to encode expected `(L, R)` labels as described above, then run `tuning_test.py`.

---

### `vision_main.py`

Production runtime that connects the vision algorithm to the UR robot via Modbus TCP.

**Responsibilities**

- Start a Modbus TCP server.
- Coordinate the two-step inspection sequence with the robot:
  1. Capture first (front) view.
  2. Capture second (back) view.
- For each view:
  - Acquire image (Pi camera or fallback path).
  - Call `detect_containers(image_bgr)` from `lid_detection.py`.
- Map view results to four container flags (C1–C4) and expose them via Modbus input registers.

**View-to-container mapping**

The mapping is implemented so that, after processing and internal swaps, the final bits correspond to the robot’s expectations for containers C1–C4. The mapping logic lives in `process_two_views` inside `vision_main.py` and uses the `(left, right)` outputs from `detect_containers` for each captured view.

Final outputs:

- `C1_RECORRECT_ADDR` (IR 131)
- `C2_RECORRECT_ADDR` (IR 132)
- `C3_RECONNECT_ADDR` (IR 133)
- `C4_RECONNECT_ADDR` (IR 134)

**Handshake (simplified)**

- Robot → server (holding registers):
  - Request new inspection.
  - Indicate which photo step is ready (1 = first view, 2 = second view).
- Server → robot (input registers):
  - Inspection ID and step completion status.
  - Results version (bumped when new c1–c4 are written).
  - Four boolean results for containers C1–C4.

**Run directly**

```bash
python3 vision_main.py
```

**Run with PM2 on the Pi (example)**

```bash
cd /home/admin/sif408_camera
PY_BIN="$(which python3)"

sudo pm2 start "$PY_BIN"   --name sif408_camera   --cwd /home/admin/sif408_camera   -- vision_main.py

sudo pm2 save
sudo pm2 startup systemd
```

---

### `simulate_main.py`

Modbus-only simulator for testing UR/PLC logic without real vision or camera.

**Behaviour**

- Starts a Modbus TCP server with the same register layout as `vision_main.py`.
- Responds to the same inspection handshake.
- When results are needed, it asks (via stdin) which containers should be flagged and writes those bits to C1–C4.

**Use cases**

- Test the UR script / PLC program that consumes the Modbus results.
- Validate the handshake and control flow independently of hardware and `lid_detection.py`.

**Run**

```bash
python3 simulate_main.py
```

---

## Typical Workflow

1. **Collect dataset**

   - Deploy `save_photos_for_calibration.py` on the Pi.
   - Capture images under real lighting and mechanical conditions into `tuning/`.
   - Rename images so their first two characters encode expected `(left, right)` results (T/F).

2. **Tune algorithm**

   - Adjust only the parameters in `lid_detection.py`.
   - Run:

     ```bash
     python3 tuning_test.py
     ```

   - Inspect:
     - Console summary for accuracy.
     - `visualised/` overlays to understand the decisions.
     - `tuning_history.csv` to track if changes improve performance.

3. **Deploy to production**

   - Copy `lid_detection.py` and `vision_main.py` to the Pi.
   - Run `vision_main.py` (optionally under PM2) so it:
     - Listens for Modbus requests from the UR.
     - Captures front/back views.
     - Computes container correction flags via `detect_containers`.
     - Publishes C1–C4 to the Modbus input registers.

4. **Test integration**

   - Use `simulate_main.py` when you only want to test the UR/PLC side.
   - Switch to `vision_main.py` for full end-to-end tests with the camera.

---

## Dependencies

Install core dependencies (example):

```bash
pip install opencv-python numpy pymodbus
```

For Raspberry Pi camera support:

- `picamera2` and its system dependencies.
- Appropriate permissions (often running under `sudo` when accessing hardware or port 502).

---

## Notes

- All threshold and ROI changes must be made in `lid_detection.py` to keep behaviour consistent across tuning, simulation, and production.
- `vision_main.py` and `simulate_main.py` focus only on orchestration, I/O, and Modbus; they do not contain any bespoke image-processing constants.
