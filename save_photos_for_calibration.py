#!/usr/bin/env python3
import os
import sys
import time
import datetime
import errno

# Terminal single-key input (no Enter needed)
import tty
import termios

SAVE_DIR = os.getenv("SAVE_DIR", "tuning")
FILENAME_PREFIX = os.getenv("FILENAME_PREFIX", "tuning")
EXT = os.getenv("EXT", "jpg").lstrip(".")

ESC = "\x1b"
CTRL_C = "\x03"
CTRL_D = "\x04"

def _mkdir_p(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def _timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

def _read_single_key():
    """Read a single keypress without requiring Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)  # raw mode: Ctrl+C arrives as '\x03' instead of SIGINT
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def main():
    try:
        from picamera2 import Picamera2
    except Exception as e:
        print(f"[ERROR] Picamera2 not available: {e}")
        print("Install picamera2 and ensure the camera is enabled, then retry.")
        sys.exit(1)

    _mkdir_p(SAVE_DIR)

    cam = Picamera2()
    cam.start()
    # Small warm-up to stabilise exposure
    time.sleep(0.5)

    print(f"[READY] Saving images to: {os.path.abspath(SAVE_DIR)}")
    print("[HELP] Press SPACE to capture; 'q', ESC, Ctrl+C or Ctrl+D to quit.\n")

    shot_count = 0
    while True:
        print("Waiting for SPACE...", end="", flush=True)
        try:
            key = _read_single_key()
        except KeyboardInterrupt:
            # In case terminal still sends SIGINT
            print("\n[EXIT] Ctrl+C")
            break
        print("")  # newline after key

        if key in (ESC, CTRL_C, CTRL_D) or key.lower() == "q":
            if key == CTRL_C:
                print("[EXIT] Ctrl+C")
            elif key == CTRL_D:
                print("[EXIT] Ctrl+D")
            elif key == ESC:
                print("[EXIT] ESC")
            else:
                print("[EXIT] q")
            break

        if key == " ":
            ts = _timestamp()
            filename = f"{FILENAME_PREFIX}_{ts}.{EXT}"
            path = os.path.join(SAVE_DIR, filename)
            try:
                cam.capture_file(path)
                shot_count += 1
                print(f"[OK] Captured #{shot_count}: {path}")
            except Exception as e:
                print(f"[ERROR] Capture failed: {e}")
        else:
            # Ignore other keys
            continue

    # Clean shutdown
    try:
        cam.stop()
    except Exception:
        pass

if __name__ == "__main__":
    main()
