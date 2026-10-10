"""Find a landing site in nadir photos and save the overlay and the decision.

    landing-predict --model runs/sdd_yolo26s/weights/best.pt --source photos/ --altitude 20 --hfov 84

The ground sampling distance comes from the flight altitude and the camera's horizontal field
of view (``--gsd`` sets it directly). Maps wider than 1536 px are reduced first: the distance
transform does not need 24 megapixels to find a 1 m disc.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from landing.classes import category_luts
from landing.visualize import overlay
from landing.zones import LandingConfig, find_site, ground_sampling_distance

MAX_WIDTH = 1536


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", type=Path, required=True, help="image or folder of images")
    parser.add_argument("--out", type=Path, default=Path("predictions"))
    parser.add_argument("--altitude", type=float, default=20.0, help="metres above ground")
    parser.add_argument("--hfov", type=float, default=84.0, help="horizontal field of view, degrees")
    parser.add_argument(
        "--gsd", type=float, help="metres per pixel of the input image (overrides altitude/hfov)"
    )
    parser.add_argument("--clearance", type=float, default=LandingConfig.clearance_m)
    parser.add_argument("--buffer", type=float, default=LandingConfig.living_buffer_m)
    parser.add_argument("--imgsz", type=int, default=1536)
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    model = YOLO(args.model)
    classes = [model.names[i] for i in sorted(model.names)]
    landable, living = category_luts(classes)
    cfg = LandingConfig(args.clearance, args.buffer)
    paths = (
        sorted(p for p in args.source.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if args.source.is_dir()
        else [args.source]
    )
    args.out.mkdir(parents=True, exist_ok=True)
    decisions = {}
    for path in paths:
        bgr = cv2.imread(str(path))
        h, w = bgr.shape[:2]
        gsd = args.gsd or ground_sampling_distance(args.altitude, args.hfov, w)
        class_map = model.predict(bgr, imgsz=args.imgsz, verbose=False)[0].semantic_mask.data.cpu().numpy()
        if w > MAX_WIDTH:
            size = (MAX_WIDTH, round(h * MAX_WIDTH / w))
            class_map = cv2.resize(class_map.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
            gsd *= w / MAX_WIDTH
        site, clearance = find_site(class_map.astype(np.int64), landable, living, gsd, cfg)
        image = overlay(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), class_map, landable, living, clearance, site, gsd, cfg
        )
        cv2.imwrite(str(args.out / f"{path.stem}_landing.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        decisions[path.name] = {**site.to_dict(), "gsd_m": gsd}
        print(path.name, decisions[path.name])
    (args.out / "decisions.json").write_text(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    main()
