"""Pick a landing site from a semantic map of the ground below a drone.

The class map becomes two masks: ground the drone can set down on, and living beings. Every
person or animal gets a buffer in metres; the distance transform of the landable ground that
remains gives, for each pixel, the radius of the largest clear disc around it, converted to
metres with the ground sampling distance (GSD). Pixels whose clearance covers the drone plus its
position error are candidates, so the whole disc the drone may touch down in keeps the buffer.
The drone takes the candidate closest to where it is now: the image centre, with the camera
pointing straight down.

Anything the camera cannot see is treated as unsafe: the image border limits the clearance.

The plan keeps a margin on top of both distances, because a segmentation boundary can be off by
a few centimetres; the safety check against a reference map does not use the margin.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class LandingConfig:
    clearance_m: float = 1.0  # drone radius plus position error
    living_buffer_m: float = 3.0  # distance kept between the landing disc and people or animals
    margin_m: float = 0.75  # planning margin for segmentation error (chosen on val); not in the safety check


@dataclass(frozen=True)
class Site:
    found: bool
    x: float = math.nan  # pixel column
    y: float = math.nan  # pixel row
    clearance_m: float = math.nan  # radius of clear landable ground around the site
    distance_m: float = math.nan  # horizontal distance from the point below the drone

    def to_dict(self) -> dict:
        return asdict(self)


def ground_sampling_distance(altitude_m: float, hfov_deg: float, width_px: int) -> float:
    """Metres per pixel for a nadir camera over flat ground."""
    return 2 * altitude_m * math.tan(math.radians(hfov_deg) / 2) / width_px


def clearance_map(landable: np.ndarray, living: np.ndarray, gsd: float, living_buffer_m: float) -> np.ndarray:
    """Metres of clear landable ground around every pixel; 0 where the drone cannot land."""
    free = landable.astype(bool)
    if living.any():  # one pixel of margin: distances between pixel centres are discrete
        free &= ndimage.distance_transform_edt(~living) * gsd > living_buffer_m + gsd
    padded = np.pad(free, 1, constant_values=False)  # outside the image is unknown
    return ndimage.distance_transform_edt(padded)[1:-1, 1:-1] * gsd


def choose_site(clearance: np.ndarray, gsd: float, cfg: LandingConfig) -> Site:
    """The candidate nearest to the image centre; ``found=False`` means hover or climb instead.
    Candidates keep the clearance plus the planning margin from anything unsafe."""
    rows, cols = np.nonzero(clearance >= cfg.clearance_m + cfg.margin_m)
    if rows.size == 0:
        return Site(found=False)
    h, w = clearance.shape
    dist = np.hypot(rows - (h - 1) / 2, cols - (w - 1) / 2)
    i = int(np.argmin(dist))
    return Site(
        True, float(cols[i]), float(rows[i]), float(clearance[rows[i], cols[i]]), float(dist[i] * gsd)
    )


def find_site(
    class_map: np.ndarray, landable_lut: np.ndarray, living_lut: np.ndarray, gsd: float, cfg: LandingConfig
) -> tuple[Site, np.ndarray]:
    """Landing site and clearance map for a map of train ids (values outside the LUT: unsafe)."""
    valid = (class_map >= 0) & (class_map < len(landable_lut))
    ids = np.where(valid, class_map, 0)
    landable = valid & landable_lut[ids]
    living = valid & living_lut[ids]
    clearance = clearance_map(landable, living, gsd, cfg.living_buffer_m + cfg.margin_m)
    return choose_site(clearance, gsd, cfg), clearance


def site_is_safe(
    site: Site,
    class_map: np.ndarray,
    landable_lut: np.ndarray,
    living_lut: np.ndarray,
    gsd: float,
    cfg: LandingConfig,
) -> bool:
    """Judge a site against a reference map: the whole disc landable, and nobody closer to the
    disc than the buffer. Pixels the reference leaves unlabelled count as unsafe."""
    h, w = class_map.shape
    rows, cols = np.ogrid[:h, :w]
    dist = np.hypot(rows - site.y, cols - site.x) * gsd
    valid = (class_map >= 0) & (class_map < len(landable_lut))
    ids = np.where(valid, class_map, 0)
    disc = dist < cfg.clearance_m  # the clearance is the distance to the nearest unsafe pixel
    if not (valid & landable_lut[ids])[disc].all():
        return False
    near = dist < cfg.clearance_m + cfg.living_buffer_m
    return not (valid & living_lut[ids])[near].any()
