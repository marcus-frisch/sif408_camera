#!/usr/bin/env python3
"""
Container lid tilt tuning tool with run history logging and visualisation.

Usage:
  - Put this script next to your `tuning/` folder and `lid_detection.py`.
  - Filenames: first char = Left (T/F), second = Right (T/F), third+ ignored.
        e.g. "TFm.jpg" -> expected (True, False)
  - Run:
        python tuning_test.py

It will:
  - Evaluate all labelled images in `tuning/`
  - Log accuracy to tuning_history.csv
  - For each processed image, save an annotated copy in `visualised/`
"""

import os
import csv
import cv2
from typing import Tuple, List, Optional
from datetime import datetime

from lid_detection import (
    detect_containers,
    detect_containers_detailed,
    LidStatus,
    ROIResult,
    # Exported parameters for logging
    ROI_TOP_FRAC,
    ROI_BOTTOM_FRAC,
    LEFT_ROI_LEFT_FRAC,
    LEFT_ROI_RIGHT_FRAC,
    RIGHT_ROI_LEFT_FRAC,
    RIGHT_ROI_RIGHT_FRAC,
    BLUR_KERNEL_SIZE,
    CANNY_LOW,
    CANNY_HIGH,
    HOUGH_RHO,
    HOUGH_THETA_DEG,
    HOUGH_THRESHOLD,
    HOUGH_MIN_LINE_LENGTH,
    HOUGH_MAX_LINE_GAP,
    ANGLE_TOLERANCE_DEG,
    ANGLE_STD_CURVED_THRESH,
    WHITE_THRESHOLD,
    WHITE_MIN_FRACTION,
    MIN_HORIZONTAL_LINES,
)

# =============================================================================
# Tuning harness config (non-vision-specific)
# =============================================================================

TUNING_DIR = "tuning"
VISUALISED_DIR = "visualised"
SAVE_VISUALISATIONS = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

PRINT_PER_IMAGE = True
HISTORY_FILE = "tuning_history.csv"
HISTORY_SHOW_LAST = 10
INCLUDE_PARAMETER_SNAPSHOT = True


# =============================================================================
# Filename parsing and image iteration
# =============================================================================

def expected_from_filename(filename: str) -> Optional[Tuple[bool, bool]]:
    """
    Decode expected (left, right) from filename.
    First char -> left, second char -> right; 'T'/'F' only.
    Extra characters ignored.
    Returns None if invalid.
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    if len(name) < 2:
        return None

    mapping = {"T": True, "F": False}
    c1 = name[0].upper()
    c2 = name[1].upper()
    if c1 not in mapping or c2 not in mapping:
        return None

    return mapping[c1], mapping[c2]


def iter_tuning_images() -> List[str]:
    if not os.path.isdir(TUNING_DIR):
        raise FileNotFoundError(
            f"Tuning directory '{TUNING_DIR}' not found. "
            f"Create it and put labelled images inside."
        )
    files: List[str] = []
    for entry in os.listdir(TUNING_DIR):
        path = os.path.join(TUNING_DIR, entry)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(entry)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            files.append(path)
    files.sort()
    return files


# =============================================================================
# Visualisation
# =============================================================================

def _save_visualisation(
    image_bgr,
    output_path: str,
    left: ROIResult,
    right: ROIResult,
    expected: Optional[Tuple[bool, bool]],
) -> None:
    """Save an annotated visualisation image."""
    vis = image_bgr.copy()

    # ROI rectangles
    def draw_box(box, needs_corr):
        x1, y1, x2, y2 = box
        color = (0, 0, 255) if needs_corr else (0, 255, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    draw_box(left.box, left.status.needs_correction)
    draw_box(right.box, right.status.needs_correction)

    # Lines inside ROIs (offset from ROI to full image coords)
    def draw_lines(lines, box, color):
        x_off, y_off = box[0], box[1]
        for x1, y1, x2, y2 in lines:
            cv2.line(vis,
                     (x1 + x_off, y1 + y_off),
                     (x2 + x_off, y2 + y_off),
                     color, 2)

    draw_lines(left.lines, left.box, (0, 255, 255))
    draw_lines(right.lines, right.box, (255, 255, 0))

    # Text overlays
    y0 = 25
    pred_txt = (
        f"Pred L={int(left.status.needs_correction)} "
        f"R={int(right.status.needs_correction)}"
    )
    cv2.putText(vis, pred_txt, (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2, cv2.LINE_AA)

    if expected is not None:
        exp_txt = f"Exp  L={int(expected[0])} R={int(expected[1])}"
        cv2.putText(vis, exp_txt, (10, y0 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)

    # Per-side summaries
    def side_summary(label: str, roi: ROIResult) -> str:
        s = roi.status
        return (
            f"{label}: "
            f"{'T' if s.needs_correction else 'F'} "
            f"{s.reason} "
            f"{s.mean_angle:.1f}deg"
        )

    left_txt = side_summary("L", left)
    right_txt = side_summary("R", right)

    cv2.putText(vis, left_txt, (10, y0 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255) if left.status.needs_correction else (0, 180, 0),
                2, cv2.LINE_AA)

    cv2.putText(vis, right_txt, (10, y0 + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255) if right.status.needs_correction else (0, 180, 0),
                2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, vis)


# =============================================================================
# History logging helpers
# =============================================================================

def _ensure_history_file() -> None:
    """Create history CSV with header if it does not exist."""
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            header = [
                "run_index",
                "timestamp",
                "total_images",
                "correct_predictions",
                "accuracy_percent",
            ]
            if INCLUDE_PARAMETER_SNAPSHOT:
                header.extend([
                    "ROI_TOP_FRAC",
                    "ROI_BOTTOM_FRAC",
                    "LEFT_ROI_LEFT_FRAC",
                    "LEFT_ROI_RIGHT_FRAC",
                    "RIGHT_ROI_LEFT_FRAC",
                    "RIGHT_ROI_RIGHT_FRAC",
                    "BLUR_KERNEL_SIZE",
                    "CANNY_LOW",
                    "CANNY_HIGH",
                    "HOUGH_RHO",
                    "HOUGH_THETA_DEG",
                    "HOUGH_THRESHOLD",
                    "HOUGH_MIN_LINE_LENGTH",
                    "HOUGH_MAX_LINE_GAP",
                    "ANGLE_TOLERANCE_DEG",
                    "ANGLE_STD_CURVED_THRESH",
                    "WHITE_THRESHOLD",
                    "WHITE_MIN_FRACTION",
                    "MIN_HORIZONTAL_LINES",
                ])
            writer.writerow(header)


def _load_history_rows() -> List[List[str]]:
    if not os.path.exists(HISTORY_FILE):
        return []
    rows: List[List[str]] = []
    with open(HISTORY_FILE, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return []
        for row in reader:
            if row:
                rows.append(row)
    return rows


def _get_next_run_index(prev_rows: List[List[str]]) -> int:
    if not prev_rows:
        return 1
    try:
        last = int(prev_rows[-1][0])
        return last + 1
    except (ValueError, IndexError):
        return len(prev_rows) + 1


def _append_history(run_index: int, total: int, correct: int, accuracy: float) -> None:
    _ensure_history_file()
    row = [
        str(run_index),
        datetime.now().isoformat(timespec="seconds"),
        str(total),
        str(correct),
        f"{accuracy:.4f}",
    ]
    if INCLUDE_PARAMETER_SNAPSHOT:
        row.extend([
            f"{ROI_TOP_FRAC}",
            f"{ROI_BOTTOM_FRAC}",
            f"{LEFT_ROI_LEFT_FRAC}",
            f"{LEFT_ROI_RIGHT_FRAC}",
            f"{RIGHT_ROI_LEFT_FRAC}",
            f"{RIGHT_ROI_RIGHT_FRAC}",
            str(BLUR_KERNEL_SIZE),
            str(CANNY_LOW),
            str(CANNY_HIGH),
            str(HOUGH_RHO),
            str(HOUGH_THETA_DEG),
            str(HOUGH_THRESHOLD),
            str(HOUGH_MIN_LINE_LENGTH),
            str(HOUGH_MAX_LINE_GAP),
            f"{ANGLE_TOLERANCE_DEG}",
            f"{ANGLE_STD_CURVED_THRESH}",
            str(WHITE_THRESHOLD),
            f"{WHITE_MIN_FRACTION}",
            str(MIN_HORIZONTAL_LINES),
        ])
    with open(HISTORY_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _print_history_summary(prev_rows: List[List[str]], current_accuracy: float) -> None:
    if not prev_rows:
        print("No previous runs logged yet.")
        return

    tail = prev_rows[-HISTORY_SHOW_LAST:]

    print(f"\nLast {len(tail)} run(s):")
    print("  run_index | timestamp           | accuracy %")
    for row in tail:
        try:
            run_idx = row[0]
            ts = row[1]
            acc = float(row[4])
            print(f"  {run_idx:>9} | {ts:19} | {acc:9.2f}")
        except (ValueError, IndexError):
            continue

    try:
        last_acc = float(prev_rows[-1][4])
        delta = current_accuracy - last_acc
        sign = "+" if delta >= 0 else "-"
        print(f"\nChange vs last run: {sign}{abs(delta):.2f}%")
    except (ValueError, IndexError):
        pass


# =============================================================================
# Main evaluation
# =============================================================================

def run_evaluation() -> None:
    image_paths = iter_tuning_images()
    if not image_paths:
        print(f"No images found in '{TUNING_DIR}'.")
        return

    if SAVE_VISUALISATIONS:
        os.makedirs(VISUALISED_DIR, exist_ok=True)

    total = 0
    correct = 0

    print("Running container lid tuning evaluation...")
    print(f"Found {len(image_paths)} image(s) in '{TUNING_DIR}'.\n")

    for path in image_paths:
        expected = expected_from_filename(path)
        if expected is None:
            if PRINT_PER_IMAGE:
                print(f"[SKIP] {os.path.basename(path)}: invalid filename pattern (needs leading T/F pair).")
            continue

        img = cv2.imread(path)
        if img is None:
            if PRINT_PER_IMAGE:
                print(f"[ERROR] {os.path.basename(path)}: could not load image.")
            continue

        # Detailed detection for visualisation
        left_res, right_res = detect_containers_detailed(img)
        predicted = (
            left_res.status.needs_correction,
            right_res.status.needs_correction,
        )
        is_ok = predicted == expected

        total += 1
        if is_ok:
            correct += 1

        if SAVE_VISUALISATIONS:
            base = os.path.basename(path)
            name, ext = os.path.splitext(base)
            vis_path = os.path.join(VISUALISED_DIR, f"{name}_vis{ext}")
            _save_visualisation(img, vis_path, left_res, right_res, expected)

        if PRINT_PER_IMAGE:
            status = "OK" if is_ok else "FAIL"
            print(f"[{status}] {os.path.basename(path)} expected={expected} predicted={predicted}")

    if total == 0:
        print("No valid labelled images to evaluate.")
        return

    accuracy = (correct / total) * 100.0

    print("\nSummary:")
    print(f"  Evaluated {total} labelled image(s)")
    print(f"  Correct predictions: {correct}")
    print(f"  Accuracy: {accuracy:.2f}%")

    prev_rows = _load_history_rows()
    run_index = _get_next_run_index(prev_rows)
    _append_history(run_index, total, correct, accuracy)
    print(f"\nRun logged as index {run_index} in '{HISTORY_FILE}'.")

    _print_history_summary(prev_rows, accuracy)


if __name__ == "__main__":
    run_evaluation()
