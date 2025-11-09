#!/usr/bin/env python3
"""
Lid detection module.

This module encapsulates the image processing and decision logic used to
determine whether the left/right containers need correction.

External API:
    detect_containers(image_bgr) -> (left_needs_correction, right_needs_correction)
    detect_containers_detailed(image_bgr) -> (left_result, right_result)

Where each result is an ROIResult containing:
    - box: (x1, y1, x2, y2) in image coordinates
    - status: LidStatus (fields documented below)
    - lines: list of line segments (x1, y1, x2, y2) in ROI coordinates
    - angles: list of angles (deg) for those segments

All tunable parameters live in this file so you can adjust behaviour in one place.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

# =============================================================================
# Tunable parameters
# =============================================================================

# Region-of-interest for the two containers (fractions of image width/height)
# Adjust these to match your mechanical layout.
ROI_TOP_FRAC = 0.33      # fraction of image height
ROI_BOTTOM_FRAC = 0.47   # fraction of image height

LEFT_ROI_LEFT_FRAC = 0.08
LEFT_ROI_RIGHT_FRAC = 0.45

RIGHT_ROI_LEFT_FRAC = 0.55
RIGHT_ROI_RIGHT_FRAC = 0.90

# Preprocessing
BLUR_KERNEL_SIZE = 5        # odd; median blur
CANNY_LOW = 15
CANNY_HIGH = 45

# Hough line detection
HOUGH_RHO = 1
HOUGH_THETA_DEG = 1.0
HOUGH_THRESHOLD = 10
HOUGH_MIN_LINE_LENGTH = 20
HOUGH_MAX_LINE_GAP = 8

# Decision thresholds
ANGLE_TOLERANCE_DEG = 1.9          # max allowed deviation from horizontal
ANGLE_STD_CURVED_THRESH = 4.0      # if std dev > this -> treat as "curved"/bad

# "No container present" / "all black" handling:
WHITE_THRESHOLD = 160              # pixel > this is "white-ish"
WHITE_MIN_FRACTION = 0.008         # min white fraction to believe a lid is present

# Require at least this many near-horizontal segments to trust the angle
MIN_HORIZONTAL_LINES = 2

# Debug flag (only used if calling code wants verbose logs)
PRINT_DEBUG_DETAILS = False


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class LidStatus:
    needs_correction: bool
    has_lid_like_content: bool
    has_top_line: bool
    mean_angle: float
    angle_std: float
    reason: str


@dataclass
class ROIResult:
    box: Tuple[int, int, int, int]                   # (x1, y1, x2, y2) in image coords
    status: LidStatus
    lines: List[Tuple[int, int, int, int]]           # line segments in ROI coords
    angles: List[float]                              # angles of those segments (deg)


# =============================================================================
# Internal helpers
# =============================================================================

def _compute_roi_box(image_bgr,
                     left_frac: float, right_frac: float,
                     top_frac: float, bottom_frac: float) -> Tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) box in pixel coords for a fractional ROI."""
    h, w = image_bgr.shape[:2]
    x1 = int(max(0.0, min(1.0, left_frac)) * w)
    x2 = int(max(0.0, min(1.0, right_frac)) * w)
    y1 = int(max(0.0, min(1.0, top_frac)) * h)
    y2 = int(max(0.0, min(1.0, bottom_frac)) * h)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid ROI fractions; check ROI_* constants.")
    return x1, y1, x2, y2


def _extract_roi(image_bgr, box: Tuple[int, int, int, int]):
    x1, y1, x2, y2 = box
    return image_bgr[y1:y2, x1:x2]


def _analyse_lid_roi(roi_bgr) -> Tuple[LidStatus, List[Tuple[int, int, int, int]], List[float]]:
    """
    Analyse one container ROI and decide if correction is needed.

    Returns:
        LidStatus,
        list of line segments used (x1, y1, x2, y2) in ROI coordinates,
        list of angles (degrees) for those segments.
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # 1) Check for sufficient bright content to qualify as a lid
    white_mask = gray > WHITE_THRESHOLD
    white_fraction = float(np.mean(white_mask))

    if white_fraction < WHITE_MIN_FRACTION:
        status = LidStatus(
            needs_correction=False,
            has_lid_like_content=False,
            has_top_line=False,
            mean_angle=0.0,
            angle_std=0.0,
            reason="no_white_content",
        )
        return status, [], []

    # 2) Edges
    blur = cv2.medianBlur(gray, BLUR_KERNEL_SIZE)
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    # 3) Hough lines
    lines = cv2.HoughLinesP(
        edges,
        rho=HOUGH_RHO,
        theta=np.deg2rad(HOUGH_THETA_DEG),
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    if lines is None:
        status = LidStatus(
            needs_correction=False,
            has_lid_like_content=True,
            has_top_line=False,
            mean_angle=0.0,
            angle_std=0.0,
            reason="no_lines_detected",
        )
        return status, [], []

    # 4) Collect near-horizontal segment angles
    used_lines: List[Tuple[int, int, int, int]] = []
    angles: List[float] = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        if abs(angle) <= 45.0:
            used_lines.append((x1, y1, x2, y2))
            angles.append(angle)

    if len(angles) < MIN_HORIZONTAL_LINES:
        status = LidStatus(
            needs_correction=False,
            has_lid_like_content=True,
            has_top_line=False,
            mean_angle=0.0,
            angle_std=0.0,
            reason="insufficient_horizontal_lines",
        )
        return status, used_lines, angles

    mean_angle = float(np.mean(angles))
    angle_std = float(np.std(angles))

    # 5) Decision rules
    if angle_std > ANGLE_STD_CURVED_THRESH:
        status = LidStatus(
            needs_correction=True,
            has_lid_like_content=True,
            has_top_line=True,
            mean_angle=mean_angle,
            angle_std=angle_std,
            reason="curved_or_noisy",
        )
    elif abs(mean_angle) > ANGLE_TOLERANCE_DEG:
        status = LidStatus(
            needs_correction=True,
            has_lid_like_content=True,
            has_top_line=True,
            mean_angle=mean_angle,
            angle_std=angle_std,
            reason="tilted",
        )
    else:
        status = LidStatus(
            needs_correction=False,
            has_lid_like_content=True,
            has_top_line=True,
            mean_angle=mean_angle,
            angle_std=angle_std,
            reason="level",
        )

    return status, used_lines, angles


# =============================================================================
# Public API
# =============================================================================

def detect_containers_detailed(image_bgr) -> Tuple[ROIResult, ROIResult]:
    """
    Run detection on a full image and return detailed results for both sides.
    """
    # Compute ROIs
    left_box = _compute_roi_box(
        image_bgr,
        LEFT_ROI_LEFT_FRAC,
        LEFT_ROI_RIGHT_FRAC,
        ROI_TOP_FRAC,
        ROI_BOTTOM_FRAC,
    )
    right_box = _compute_roi_box(
        image_bgr,
        RIGHT_ROI_LEFT_FRAC,
        RIGHT_ROI_RIGHT_FRAC,
        ROI_TOP_FRAC,
        ROI_BOTTOM_FRAC,
    )

    # Extract ROIs
    left_roi = _extract_roi(image_bgr, left_box)
    right_roi = _extract_roi(image_bgr, right_box)

    # Analyse
    left_status, left_lines, left_angles = _analyse_lid_roi(left_roi)
    right_status, right_lines, right_angles = _analyse_lid_roi(right_roi)

    if PRINT_DEBUG_DETAILS:
        print(
            f"Left: need={left_status.needs_correction}, "
            f"white={left_status.has_lid_like_content}, top={left_status.has_top_line}, "
            f"angle={left_status.mean_angle:.2f}, std={left_status.angle_std:.2f}, "
            f"reason={left_status.reason}"
        )
        print(
            f"Right: need={right_status.needs_correction}, "
            f"white={right_status.has_lid_like_content}, top={right_status.has_top_line}, "
            f"angle={right_status.mean_angle:.2f}, std={right_status.angle_std:.2f}, "
            f"reason={right_status.reason}"
        )

    left_result = ROIResult(
        box=left_box,
        status=left_status,
        lines=left_lines,
        angles=left_angles,
    )

    right_result = ROIResult(
        box=right_box,
        status=right_status,
        lines=right_lines,
        angles=right_angles,
    )

    return left_result, right_result


def detect_containers(image_bgr) -> Tuple[bool, bool]:
    """
    Simplified API for production use.

    Returns:
        (left_needs_correction, right_needs_correction)
    """
    left, right = detect_containers_detailed(image_bgr)
    return left.status.needs_correction, right.status.needs_correction
