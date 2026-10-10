"""Score a segmentation model and the landing decisions built on it, on held-out images.

    landing-eval --model runs/sdd_yolo26s/weights/best.pt --data data/landing/data.yaml --split test

Segmentation: IoU per class, plus IoU of the two masks the landing rule uses (landable ground;
people and animals). Landing: on every image, the site each policy picks is judged against the
hand-labelled mask.

- straight-down: descend where the drone is, without a camera;
- nearest-ground: the predicted landable pixel nearest the centre (no clearance, no buffer);
- clearance, margin m: this repo's rule, a clear disc and a buffer around living beings, planned
  with a margin m for segmentation error.

A site is safe if, in the labelled mask, the disc of radius ``clearance_m`` around it is landable
and nobody is closer to the disc than the buffer. Abstaining (no site found) is correct when the labelled mask
has no site either.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from landing.classes import category_luts
from landing.zones import LandingConfig, Site, find_site, site_is_safe

log = logging.getLogger("landing.evaluate")
BASELINES = ("straight-down", "nearest-ground")


def load_reference(path: Path, mapping: dict[int, int | str]) -> np.ndarray:
    """Raw-id mask -> train ids, -1 where ignored."""
    lut = np.full(256, -1, dtype=np.int16)
    for raw, train in mapping.items():
        if isinstance(train, int) and 0 <= int(raw) < 256:
            lut[int(raw)] = train
    return lut[np.array(Image.open(path))]


def confusion(pred: np.ndarray, ref: np.ndarray, n: int) -> np.ndarray:
    keep = ref >= 0
    return np.bincount(n * ref[keep].astype(np.int64) + pred[keep], minlength=n * n).reshape(n, n)


def iou(cm: np.ndarray) -> np.ndarray:
    inter = np.diag(cm).astype(float)
    union = cm.sum(0) + cm.sum(1) - inter
    return np.where(union > 0, inter / np.maximum(union, 1), np.nan)


def binary_iou(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray) -> tuple[int, int]:
    return int((pred & ref & valid).sum()), int(((pred | ref) & valid).sum())


def decide(policy: str, pred: np.ndarray, landable, living, gsd: float, cfg: LandingConfig) -> Site:
    h, w = pred.shape
    if policy == "straight-down":
        return Site(True, (w - 1) / 2, (h - 1) / 2, 0.0, 0.0)
    if policy == "nearest-ground":
        return find_site(
            pred, landable, living, gsd, LandingConfig(clearance_m=gsd / 2, living_buffer_m=0.0, margin_m=0.0)
        )[0]
    margin = float(policy.split("margin ")[1].split(" m")[0])  # "clearance, margin 0.5 m"
    return find_site(pred, landable, living, gsd, replace(cfg, margin_m=margin))[0]


def evaluate(
    model_path: str,
    data_yaml: Path,
    split: str,
    imgsz: int,
    gsd: float,
    cfg: LandingConfig,
    margins: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict:
    from ultralytics import YOLO

    data = yaml.safe_load(data_yaml.read_text())
    root = Path(data["path"])
    classes = [data["names"][i] for i in sorted(data["names"])]
    landable, living = category_luts(classes)
    model = YOLO(model_path)
    n = len(classes)
    cm = np.zeros((n, n), dtype=np.int64)
    masks = {"landable": [0, 0], "living": [0, 0]}
    policies = [*BASELINES, *(f"clearance, margin {m:g} m" for m in margins)]
    outcomes = {p: [] for p in policies}
    oracle, seconds = [], []
    images = sorted((root / data[split]).glob("*.jpg"))
    for path in images:
        ref = load_reference(
            root / data.get("masks_dir", "masks") / split / f"{path.stem}.png", data["label_mapping"]
        )
        started = time.perf_counter()
        result = model.predict(str(path), imgsz=imgsz, verbose=False)[0]
        seconds.append(time.perf_counter() - started)
        pred = result.semantic_mask.data.cpu().numpy().astype(np.int64)
        if pred.shape != ref.shape:
            pred = np.array(
                Image.fromarray(pred.astype(np.uint8)).resize(ref.shape[::-1], Image.NEAREST), np.int64
            )
        cm += confusion(pred, ref, n)
        valid = ref >= 0
        ref_ids = np.where(valid, ref, 0)
        for name, lut in (("landable", landable), ("living", living)):
            i, u = binary_iou(lut[pred], lut[ref_ids] & valid, valid)
            masks[name][0] += i
            masks[name][1] += u
        ref_site = find_site(ref, landable, living, gsd, replace(cfg, margin_m=0.0))[0]
        oracle.append(ref_site.found)
        for policy in policies:
            site = decide(policy, pred, landable, living, gsd, cfg)
            if not site.found:
                outcomes[policy].append(("abstain", None))
            else:
                safe = site_is_safe(site, ref, landable, living, gsd, cfg)
                outcomes[policy].append(("safe" if safe else "unsafe", site.distance_m))

    per_class = iou(cm)
    n_img = len(images)
    landing = {}
    for policy, results in outcomes.items():
        kinds = [k for k, _ in results]
        landing[policy] = {
            "safe": kinds.count("safe") / n_img,
            "unsafe": kinds.count("unsafe") / n_img,
            "abstain": kinds.count("abstain") / n_img,
            "abstain_correct": sum(k == "abstain" and not o for k, o in zip(kinds, oracle, strict=True))
            / n_img,
            "median_distance_m": float(np.median([d for k, d in results if k != "abstain"] or [np.nan])),
        }
    return {
        "model": model_path,
        "split": split,
        "images": n_img,
        "gsd_m": gsd,
        "landing_config": cfg.__dict__,
        "miou": float(np.nanmean(per_class)),
        "iou": {c: float(v) for c, v in zip(classes, per_class, strict=True)},
        "iou_landable": masks["landable"][0] / max(masks["landable"][1], 1),
        "iou_living": masks["living"][0] / max(masks["living"][1], 1),
        "reference_has_site": float(np.mean(oracle)),
        "landing": landing,
        "seconds_per_image": float(np.median(seconds[1:] if len(seconds) > 1 else seconds)),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument(
        "--gsd", type=float, default=0.02, help="metres per pixel at the 1536 px working width"
    )
    parser.add_argument("--clearance", type=float, default=LandingConfig.clearance_m)
    parser.add_argument("--buffer", type=float, default=LandingConfig.living_buffer_m)
    parser.add_argument("--margins", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = evaluate(
        args.model,
        args.data,
        args.split,
        args.imgsz,
        args.gsd,
        LandingConfig(args.clearance, args.buffer),
        tuple(args.margins),
    )
    out = args.out or Path(args.model).resolve().parents[1] / f"eval_{args.split}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "iou"}, indent=2))
    print("saved to", out)


if __name__ == "__main__":
    main()
