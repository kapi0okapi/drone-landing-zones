"""Overlay of the landing decision: landable ground, no-go areas, the chosen site."""

from __future__ import annotations

import cv2
import numpy as np

from landing.zones import LandingConfig, Site

GREEN = (46, 160, 67)
RED = (214, 69, 65)
MAGENTA = (190, 60, 190)
ORANGE = (255, 79, 26)


def overlay(
    rgb: np.ndarray,
    class_map: np.ndarray,
    landable_lut: np.ndarray,
    living_lut: np.ndarray,
    clearance: np.ndarray,
    site: Site,
    gsd: float,
    cfg: LandingConfig,
    alpha: float = 0.4,
) -> np.ndarray:
    """RGB image with ground tinted green (landable) or red (not), people magenta, the clearance
    candidates outlined, the drone's position (cross) and the chosen site (ring + line)."""
    h, w = class_map.shape
    if rgb.shape[:2] != (h, w):
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    ids = np.clip(class_map, 0, len(landable_lut) - 1)
    colour = np.where(landable_lut[ids][..., None], GREEN, RED).astype(np.uint8)
    colour[living_lut[ids]] = MAGENTA
    out = cv2.addWeighted(rgb, 1 - alpha, colour, alpha, 0)

    candidates = (clearance >= cfg.clearance_m).astype(np.uint8)
    contours, _ = cv2.findContours(candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), max(1, w // 600))

    thick = max(2, w // 400)
    centre = (w // 2, h // 2)
    arm = w // 60
    cv2.line(out, (centre[0] - arm, centre[1]), (centre[0] + arm, centre[1]), (255, 255, 255), thick)
    cv2.line(out, (centre[0], centre[1] - arm), (centre[0], centre[1] + arm), (255, 255, 255), thick)
    if site.found:
        at = (round(site.x), round(site.y))
        cv2.line(out, centre, at, ORANGE, thick, cv2.LINE_AA)
        cv2.circle(out, at, max(3, round(cfg.clearance_m / gsd)), ORANGE, thick, cv2.LINE_AA)
        label = f"land here: {site.distance_m:.1f} m away, {site.clearance_m:.1f} m clear"
    else:
        label = "no safe site: hold position"
    scale = w / 1600
    org = (round(16 * scale), round(44 * scale))
    cv2.putText(
        out, label, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), round(5 * scale) + 1, cv2.LINE_AA
    )
    cv2.putText(
        out, label, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), round(2 * scale) + 1, cv2.LINE_AA
    )
    return out
