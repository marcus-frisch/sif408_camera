#!/usr/bin/env python3
"""
Camera Inspection Modbus TCP Server (UR3-synced)

Refactored to delegate all image processing to `lid_detection.py`.

Pipeline summary:

  - UR requests an inspection via Modbus.
  - This server captures two views: "front" and "back".
      * Front view: containers C1 (left), C3 (right)
      * Back view:  containers C2 (left), C4 (right)
  - Each view is passed to `lid_detection.detect_containers(image_bgr)`.
      * That returns (left_needs_correction, right_needs_correction).
  - Combined boolean results are written to Modbus input registers:
      * c1_recorrect, c2_recorrect, c3_reconnect, c4_reconnect

Modbus behaviour and handshake are kept identical to the original version.
"""

import os
import sys
import time
import logging
import threading
import cv2
from datetime import datetime
from typing import Dict, Any

from pymodbus.datastore import (
    ModbusServerContext, ModbusSlaveContext, ModbusSequentialDataBlock
)

# Import the tuned lid detection (no Modbus / no camera inside)
from lid_detection import detect_containers

# ---------------------------- Configuration ---------------------------------

GUI_ENABLED = False
USE_PI_CAMERA = os.getenv("USE_PI_CAMERA", "1") not in ("0", "false", "False")
SAVE_DIR = os.getenv("SAVE_DIR", "imgs")
MODBUS_PORT = int(os.getenv("MODBUS_PORT", "502"))

# Fallback image files if Pi Camera is disabled/unavailable
IMAGE_FRONT_PATH = os.getenv("IMAGE_FRONT_PATH", "sample_front.jpg")
IMAGE_BACK_PATH = os.getenv("IMAGE_BACK_PATH", "sample_back.jpg")

# ---------------------------- Camera setup ----------------------------------

camera = None
if USE_PI_CAMERA:
    try:
        from picamera2 import Picamera2
        print("[CONFIG] Pi Camera enabled")
        camera = Picamera2()
        camera.start()
    except Exception as e:
        print(f"[CONFIG] WARNING: Pi Camera unavailable ({e}). Using file paths.")
        USE_PI_CAMERA = False
else:
    print("[CONFIG] Pi Camera disabled (USE_PI_CAMERA=0)")

# ---------------------------- Modbus setup ----------------------------------

# Addressing exactly as in the UR script

INSPECTION_ID_ADDR = 128   # IR
PHOTO_STEP_DONE_ADDR = 129 # IR
RESULTS_VERSION_ADDR = 130 # IR

C1_RECORRECT_ADDR = 131    # IR
C2_RECORRECT_ADDR = 132    # IR
C3_RECONNECT_ADDR = 133    # IR  (UR name: c3_reconnect)
C4_RECONNECT_ADDR = 134    # IR  (UR name: c4_reconnect)

MM_RECEIVED_INSTRUCTION_ADDR = 135  # HR
PHOTO_READY_STEP_ADDR = 136         # HR

# Shared datastore
_hr_block = ModbusSequentialDataBlock(0, [0] * 200)
_ir_block = ModbusSequentialDataBlock(0, [0] * 200)

_ctx_unit = ModbusSlaveContext(hr=_hr_block, ir=_ir_block, di=None, co=None)
slaves = {
    0xFF: _ctx_unit,  # UR script uses unit-id 255
    0x01: _ctx_unit,  # alternative
    0x00: _ctx_unit,  # catch-all
}
context = ModbusServerContext(slaves=slaves, single=False)

# Thread-safety for datastore access
_context_lock = threading.Lock()


def _hr_get(addr: int, count: int = 1):
    with _context_lock:
        return context[0xFF].getValues(3, addr, count=count)  # 3 = HR


def _hr_set(addr: int, values):
    with _context_lock:
        context[0xFF].setValues(3, addr, values)


def _ir_get(addr: int, count: int = 1):
    with _context_lock:
        return context[0xFF].getValues(4, addr, count=count)  # 4 = IR


def _ir_set(addr: int, values):
    with _context_lock:
        context[0xFF].setValues(4, addr, values)

# --------------------------- Utility: save path ------------------------------


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _photo_path(kind: str, inspection_id: int) -> str:
    """
    Build a unique filename for captured photos.
    kind: 'first_view' or 'second_view'
    """
    _ensure_dir(SAVE_DIR)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"inspection_{inspection_id}_{kind}_{ts}.jpg"
    return os.path.join(SAVE_DIR, fname)

# --------------------------- Capture (async) ---------------------------------


def take_photo_async(kind: str, inspection_id: int) -> Dict[str, Any]:
    """
    Capture a photo asynchronously.

    Returns a dict:
        {'path': str or None, 'done': bool}
    kind: "first" or "second"
    """
    result = {'path': None, 'done': False}

    def _capture():
        try:
            if USE_PI_CAMERA and camera is not None:
                tag = "first_view" if kind == "first" else "second_view"
                save_path = _photo_path(tag, inspection_id)
                print(f"[CAMERA] Capturing {tag.replace('_', ' ').title()} from Pi camera...")
                camera.capture_file(save_path)
                print(f"[CAMERA] Saved to: {save_path}")
                result['path'] = save_path
            else:
                # Fallback to static sample images when Pi camera is disabled/unavailable
                result['path'] = IMAGE_FRONT_PATH if kind == "first" else IMAGE_BACK_PATH

            result['done'] = True

        except Exception as e:
            print(f"[CAMERA] ERROR during capture: {e}")
            result['done'] = True
            result['path'] = None

    t = threading.Thread(target=_capture, daemon=True)
    t.start()
    return result

# ------------------------ Image processing adapter ---------------------------


def _load_image(path: str, label: str):
    """
    Load an image for processing; log on failure.
    """
    if not path:
        print(f"[AUTO DETECT] ERROR: No path provided for {label} view.")
        return None

    img = cv2.imread(path)
    if img is None:
        print(f"[AUTO DETECT] ERROR: Failed to load {label} view image from {path}.")
    return img


def process_two_views(front_path: str, back_path: str):
    """
    Process both camera views using lid_detection.detect_containers.

    Mapping (after fixes):
      - Front view:
          left  -> C3
          right -> C1
      - Back view:
          left  -> C2
          right -> C4

    Then swap C1 and C4 indexing to match external expectations:
      final_C1 = mapped_C4
      final_C4 = mapped_C1

    Returns:
        dict: {'c1': int, 'c2': int, 'c3': int, 'c4': int}
              where 1 = needs recorrection, 0 = OK
    """
    # Default all to OK
    c1 = c2 = c3 = c4 = 0

    # Front view: C3 (left), C1 (right)
    front_img = _load_image(front_path, "front")
    if front_img is not None:
        f_left, f_right = detect_containers(front_img)
        c3 = 1 if f_left else 0
        c1 = 1 if f_right else 0
        print(f"[AUTO DETECT] Front view (pre-swap) -> C3={c3}, C1={c1}")

    # Back view: C2 (left), C4 (right)
    back_img = _load_image(back_path, "back")
    if back_img is not None:
        b_left, b_right = detect_containers(back_img)
        c2 = 1 if b_left else 0
        c4 = 1 if b_right else 0
        print(f"[AUTO DETECT] Back view  (pre-swap) -> C2={c2}, C4={c4}")

    # Swap C1 and C4 indexing
    c1, c4 = c4, c1
    print(f"[AUTO DETECT] Post-swap      -> C1={c1}, C2={c2}, C3={c3}, C4={c4}")

    results = {'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4}
    print(f"[AUTO DETECT] Combined results: {results}")
    return results

# --------------------------- Error formatting --------------------------------

class PymodbusErrorFilter(logging.Filter):
    """
    Suppress the noisy 'unpack requires a buffer of 4 bytes' error from pymodbus
    and print a clearer hint instead.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if (
            "unpack requires a buffer of 4 bytes" in msg
            and "Unknown exception" in msg
            and "stream server forcing disconnect" in msg
        ):
            print("Error connecting to UR: Teach pendent > Installation > Fieldbus > MODBUS > Refresh List... Registers should then show green on teach pendent.")
            return False  # block the original log line
        return True

# --------------------------- Logic loop --------------------------------------


def inspection_loop():
    inspection_id = 0
    photo_step_done = 0   # 0 none, 1 = first view captured, 2 = both views processed
    results_version = 0

    prev_mm = 0  # for rising-edge detect on HR135

    front_cap = None
    back_cap = None
    front_path = None
    back_path = None

    # Initial publish
    _ir_set(INSPECTION_ID_ADDR, [inspection_id])
    _ir_set(PHOTO_STEP_DONE_ADDR, [photo_step_done])
    _ir_set(RESULTS_VERSION_ADDR, [results_version])
    _ir_set(C1_RECORRECT_ADDR, [0])
    _ir_set(C2_RECORRECT_ADDR, [0])
    _ir_set(C3_RECONNECT_ADDR, [0])
    _ir_set(C4_RECONNECT_ADDR, [0])

    print("[CAMERA] Inspection loop started")
    print(f"[CAMERA] Mode: {'GUI' if GUI_ENABLED else 'Automated CV'}")
    print(f"[CAMERA] Camera: {'Pi Camera' if USE_PI_CAMERA else 'File images'}")

    while True:
        try:
            mm = _hr_get(MM_RECEIVED_INSTRUCTION_ADDR, 1)[0]
            step = _hr_get(PHOTO_READY_STEP_ADDR, 1)[0]

            # Rising edge on mm_recv_inst -> begin new inspection
            if mm == 1 and prev_mm == 0:
                inspection_id += 1
                photo_step_done = 0
                front_cap = back_cap = None
                front_path = back_path = None

                print(f"\n[CAMERA] New inspection requested. ID = {inspection_id}\n")

                _ir_set(INSPECTION_ID_ADDR, [inspection_id])
                _ir_set(PHOTO_STEP_DONE_ADDR, [photo_step_done])
                # UR program is responsible for clearing HR135.

            prev_mm = mm

            # Step 1: First (front) view requested
            if step == 1 and photo_step_done == 0 and front_cap is None:
                print("[CAMERA] First view requested.")
                print("[CAMERA] This photo shows: C1 (left), C3 (right)")
                front_cap = take_photo_async("first", inspection_id)

            # When first capture completes
            if front_cap is not None and front_cap.get('done') and front_path is None:
                front_path = front_cap.get('path')
                photo_step_done = 1
                _ir_set(PHOTO_STEP_DONE_ADDR, [1])  # IR129=1
                print(f"[CAMERA] First view complete: {front_path}")

            # Step 2: Second (back) view requested
            if step == 2 and photo_step_done == 1 and back_cap is None:
                print("[CAMERA] Second view requested.")
                print("[CAMERA] This photo shows: C2 (left), C4 (right)")
                back_cap = take_photo_async("second", inspection_id)

            # When second capture completes -> run CV and publish
            if (
                back_cap is not None
                and back_cap.get('done')
                and back_path is None
                and photo_step_done == 1
            ):
                back_path = back_cap.get('path')

                # Process both views via lid_detection
                results = process_two_views(front_path, back_path)

                c1 = int(results.get("c1", 0))
                c2 = int(results.get("c2", 0))
                c3 = int(results.get("c3", 0))
                c4 = int(results.get("c4", 0))

                # Publish results to Input Registers
                _ir_set(C1_RECORRECT_ADDR, [c1])
                _ir_set(C2_RECORRECT_ADDR, [c2])
                _ir_set(C3_RECONNECT_ADDR, [c3])  # keep UR naming
                _ir_set(C4_RECONNECT_ADDR, [c4])

                photo_step_done = 2
                _ir_set(PHOTO_STEP_DONE_ADDR, [2])  # IR129=2

                results_version += 1
                _ir_set(RESULTS_VERSION_ADDR, [results_version])

                print(f"[CAMERA] Second view complete; c1..c4 = {(c1, c2, c3, c4)}")
                print(f"[CAMERA] Results version bumped to {results_version}")

            time.sleep(0.10)  # ~10Hz

        except Exception as e:
            print(f"[LOOP] ERROR: {e}")
            time.sleep(0.25)

# --------------------------- Server runner -----------------------------------


def _start_modbus_server(context, host: str, port: int):
    """
    Start Modbus TCP server (compatible with pymodbus 2.x / 3.x variants).
    """
    try:
        from pymodbus.server import StartTcpServer as _Start
        _Start(context=context, address=(host, port))
        return
    except Exception:
        pass

    try:
        from pymodbus.server.sync import StartTcpServer as _StartSync
        _StartSync(context=context, address=(host, port))
        return
    except Exception:
        pass

    from pymodbus.server import ModbusTcpServer
    srv = ModbusTcpServer(context, address=(host, port), defer_start=False)
    srv.serve_forever()


def run_modbus_server():
    if os.getenv("DEBUG_MODBUS"):
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("pymodbus").setLevel(logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("pymodbus").setLevel(logging.INFO)

    # Attach filter to both the main pymodbus logger and the sub-logger
    err_filter = PymodbusErrorFilter()
    logging.getLogger("pymodbus").addFilter(err_filter)
    logging.getLogger("pymodbus.logging").addFilter(err_filter)

    print(f"[MODBUS] Starting server on port {MODBUS_PORT}")
    _start_modbus_server(context, host="0.0.0.0", port=MODBUS_PORT)


# ------------------------------ Main -----------------------------------------


def main():
    if GUI_ENABLED:
        print("[MAIN] GUI mode not implemented; set GUI_ENABLED=False.")
        sys.exit(1)
    else:
        logic = threading.Thread(target=inspection_loop, daemon=True)
        logic.start()

        print("[MAIN] Automated mode: Running Modbus server")
        print("[MAIN] Press Ctrl+C to exit")
        try:
            run_modbus_server()
        except KeyboardInterrupt:
            print("\n[MAIN] Shutting down...")
            try:
                if camera is not None:
                    camera.stop()
            except Exception:
                pass
            sys.exit(0)


if __name__ == "__main__":
    main()
