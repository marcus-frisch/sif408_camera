#!/usr/bin/env python3
"""
Camera Inspection Modbus TCP Server

Implements the following state machine:

- Continuously publishes:
    inspection_id
    photo_step_done
    results_version
    c1_recorrect
    c2_recorrect
    c3_recorrect
    c4_recorrect

- Triggers a new inspection when the robot sets mm_received_instruction = 1
- Waits for photo_ready_step == 1 to mark first view done (after 3 s)
- Waits for photo_ready_step == 2 to:
    - wait 3 s
    - run ProcessContainers() (prompt terminal input)
    - publish the 4 correction bits atomically by bumping results_version
"""

import threading
import time
from pymodbus.server import StartTcpServer
from pymodbus.datastore import (
    ModbusSlaveContext,
    ModbusServerContext,
    ModbusSequentialDataBlock,
)

# ---------------------------------------------------------------------------
# Modbus address map
# Robot -> server (we READ these from holding registers)
MM_RECEIVED_INSTRUCTION_ADDR = 135   # robot writes 1 to start new inspection
PHOTO_READY_STEP_ADDR = 136          # robot writes 1 or 2

# Server -> robot (we WRITE these to input registers so robot can read)
INSPECTION_ID_ADDR = 128
PHOTO_STEP_DONE_ADDR = 129
RESULTS_VERSION_ADDR = 130
C1_RECORRECT_ADDR = 131
C2_RECORRECT_ADDR = 132
C3_RECORRECT_ADDR = 133
C4_RECORRECT_ADDR = 134
# ---------------------------------------------------------------------------

# Create data store with enough space
store = ModbusSlaveContext(
    hr=ModbusSequentialDataBlock(0, [0] * 200),  # holding registers 0..199
    ir=ModbusSequentialDataBlock(0, [0] * 200),  # input registers 0..199
    di=ModbusSequentialDataBlock(0, [0] * 200),
    co=ModbusSequentialDataBlock(0, [0] * 200),
)
context = ModbusServerContext(slaves=store, single=True)


# ---------------------------------------------------------------------------
# Helpers to read robot-driven values (holding registers, fc=3)
def _hr_get(addr: int, count: int = 1):
    slave_id = 0x00
    return context[slave_id].getValues(3, addr, count=count)


def read_mm_received_instruction() -> int:
    return _hr_get(MM_RECEIVED_INSTRUCTION_ADDR, 1)[0]


def read_photo_ready_step() -> int:
    return _hr_get(PHOTO_READY_STEP_ADDR, 1)[0]


# ---------------------------------------------------------------------------
# Helpers to publish to robot (input registers, fc=4)
def _ir_set(addr: int, values):
    slave_id = 0x00
    context[slave_id].setValues(4, addr, values)


def publish_inspection_state(
    inspection_id: int,
    photo_step_done: int,
    results_version: int,
    c1: bool,
    c2: bool,
    c3: bool,
    c4: bool,
):
    _ir_set(INSPECTION_ID_ADDR, [inspection_id])
    _ir_set(PHOTO_STEP_DONE_ADDR, [photo_step_done])
    _ir_set(C1_RECORRECT_ADDR, [1 if c1 else 0])
    _ir_set(C2_RECORRECT_ADDR, [1 if c2 else 0])
    _ir_set(C3_RECORRECT_ADDR, [1 if c3 else 0])
    _ir_set(C4_RECORRECT_ADDR, [1 if c4 else 0])
    _ir_set(RESULTS_VERSION_ADDR, [results_version])


# ---------------------------------------------------------------------------
def process_containers():
    """
    Prompt for a string. User may type digits 1..4 in any combination, e.g.:
        ""    -> [False, False, False, False]
        "1"   -> [True,  False, False, False]
        "12"  -> [True,  True,  False, False]
        "234" -> [False, True,  True,  True]
    Returns tuple (c1, c2, c3, c4).
    """
    s = input("Enter containers needing correction (e.g. '12', '3', '14'): ").strip()
    c1 = "1" in s
    c2 = "2" in s
    c3 = "3" in s
    c4 = "4" in s
    return c1, c2, c3, c4


# ---------------------------------------------------------------------------
def inspection_loop():
    """
    Runs the main logic at ~10 Hz, matching the pseudocode.
    """
    inspection_id = 0
    photo_step_done = 0   # 0 none, 1 first view, 2 second view
    results_version = 0
    c1_recorrect = False
    c2_recorrect = False
    c3_recorrect = False
    c4_recorrect = False

    print("[CAMERA] Inspection loop started.")

    while True:
        # ---- CONTINUOUS PUBLISH ----
        publish_inspection_state(
            inspection_id,
            photo_step_done,
            results_version,
            c1_recorrect,
            c2_recorrect,
            c3_recorrect,
            c4_recorrect,
        )

        # ---- START NEW INSPECTION ----
        if read_mm_received_instruction() == 1:
            inspection_id += 1
            photo_step_done = 0
            # Keep previous cX_recorrect values until next commit
            print(f"[CAMERA] New inspection requested. ID = {inspection_id}")

            # Clear the trigger so we don't re-trigger next cycle
            # (UR can also clear it its side; this is defensive.)
            _ir_set(MM_RECEIVED_INSTRUCTION_ADDR, [0])

        # ---- FIRST VIEW ----
        photo_ready_step = read_photo_ready_step()
        if photo_ready_step == 1 and photo_step_done == 0:
            print("[CAMERA] First view ready, waiting 3 s to simulate capture...")
            time.sleep(3.0)
            photo_step_done = 1
            publish_inspection_state(
                inspection_id,
                photo_step_done,
                results_version,
                c1_recorrect,
                c2_recorrect,
                c3_recorrect,
                c4_recorrect,
            )
            print("[CAMERA] First view complete.")

        # ---- SECOND VIEW + ATOMIC COMMIT ----
        photo_ready_step = read_photo_ready_step()  # re-read in case it changed
        if photo_ready_step == 2 and photo_step_done == 1:
            print("[CAMERA] Second view ready, waiting 3 s to simulate capture...")
            time.sleep(3.0)

            # This is where real image processing would happen.
            new_c1, new_c2, new_c3, new_c4 = process_containers()

            # Atomic commit: write bits first, then bump version
            c1_recorrect = new_c1
            c2_recorrect = new_c2
            c3_recorrect = new_c3
            c4_recorrect = new_c4

            
            photo_step_done = 2
            results_version += 1   # commit point

            publish_inspection_state(
                inspection_id,
                photo_step_done,
                results_version,
                c1_recorrect,
                c2_recorrect,
                c3_recorrect,
                c4_recorrect,
            )

            print(
                f"[CAMERA] Results committed (version {results_version}): "
                f"C1={c1_recorrect}, C2={c2_recorrect}, "
                f"C3={c3_recorrect}, C4={c4_recorrect}"
            )

        # 10 Hz
        time.sleep(0.1)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Start logic thread
    logic_thread = threading.Thread(target=inspection_loop, daemon=True)
    logic_thread.start()

    # Start Modbus TCP server
    # Use 1502 to avoid root; change to 502 if you run with sudo.
    print("[MODBUS] Starting server")
    StartTcpServer(context=context, address=("0.0.0.0", 502))
