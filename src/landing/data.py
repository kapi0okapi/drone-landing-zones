"""Semantic Drone Dataset -> Ultralytics semantic layout at a working resolution.

    landing-prepare --raw data/SemanticDroneDatasetPublic --out data/landing

The 400 public images (6000 x 4000 px, nadir, 5-30 m above ground) are resized to 1536 x 1024
and their colour-coded masks decoded to class ids. Consecutive images show the same house, so the
split assigns blocks of 10 consecutive images, not single images, to train / val / test: a test
image never has a near-duplicate neighbour in the training set.

Output: images/<split>/*.jpg, masks/<split>/*.png (raw class ids), data.yaml, split.csv.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from landing.classes import label_mapping, train_classes

log = logging.getLogger("landing.data")
SIZE = (1536, 1024)
SPLITS = {"train": 0.7, "val": 0.1, "test": 0.2}


def read_class_dict(path: Path) -> tuple[list[str], np.ndarray]:
    """Class names and their RGB mask colours, in file order (= raw id)."""
    with open(path, newline="") as f:
        rows = [{k.strip().lower(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]
    names = [r["name"] for r in rows]
    colours = np.array([[int(r["r"]), int(r["g"]), int(r["b"])] for r in rows], dtype=np.uint8)
    return names, colours


def decode_rgb(mask: np.ndarray, colours: np.ndarray, unknown: int = 0) -> np.ndarray:
    """Colour-coded (H, W, 3) mask -> raw class ids; colours not in the table get ``unknown``."""
    key = lambda rgb: (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2]  # noqa: E731
    table = key(colours)
    order = np.argsort(table)
    flat = key(mask)
    pos = np.clip(np.searchsorted(table[order], flat), 0, len(table) - 1)
    ids = order[pos]
    return np.where(table[ids] == flat, ids, unknown).astype(np.uint8)


def block_split(stems: list[str], block: int = 10, seed: int = 0) -> dict[str, str]:
    """Stem -> split; blocks of consecutive images stay together."""
    blocks = [stems[i : i + block] for i in range(0, len(stems), block)]
    order = np.random.default_rng(seed).permutation(len(blocks))
    n_train = round(len(blocks) * SPLITS["train"])
    n_val = round(len(blocks) * SPLITS["val"])
    out = {}
    for rank, b in enumerate(order):
        split = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"
        out |= {s: split for s in blocks[b]}
    return out


def find_layout(raw: Path) -> tuple[Path, Path, Path]:
    """class_dict.csv, the image folder and the colour mask folder, wherever the archive put them."""
    class_dict = next(iter(sorted(raw.rglob("class_dict*.csv"))))
    images = max((p.parent for p in raw.rglob("*.jpg")), key=lambda d: len(list(d.glob("*.jpg"))))
    masks = [d for d in {p.parent for p in raw.rglob("*.png")} if "label" in str(d).lower()]
    if not masks:
        raise FileNotFoundError(f"no label PNG folder under {raw}")
    return class_dict, images, max(masks, key=lambda d: len(list(d.glob("*.png"))))


def prepare(raw: Path, out: Path, size: tuple[int, int] = SIZE, seed: int = 0) -> Path:
    class_dict, image_dir, mask_dir = find_layout(raw)
    names, colours = read_class_dict(class_dict)
    stems = sorted(p.stem for p in image_dir.glob("*.jpg") if (mask_dir / f"{p.stem}.png").exists())
    log.info("%d image/mask pairs, %d classes (%s)", len(stems), len(names), class_dict)
    splits = block_split(stems, seed=seed)
    for stem in stems:
        split = splits[stem]
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "masks" / split).mkdir(parents=True, exist_ok=True)
        Image.open(image_dir / f"{stem}.jpg").convert("RGB").resize(size, Image.LANCZOS).save(
            out / "images" / split / f"{stem}.jpg", quality=95
        )
        mask = Image.open(mask_dir / f"{stem}.png")
        if mask.mode in ("L", "P"):
            ids = np.array(mask.resize(size, Image.NEAREST), dtype=np.uint8)
        else:
            ids = decode_rgb(np.array(mask.convert("RGB").resize(size, Image.NEAREST)), colours)
        Image.fromarray(ids).save(out / "masks" / split / f"{stem}.png")
    with open(out / "split.csv", "w", newline="") as f:
        csv.writer(f).writerows([("image", "split"), *sorted(splits.items())])
    data = {
        "path": str(out.resolve()),
        **{s: f"images/{s}" for s in SPLITS},
        "masks_dir": "masks",
        "names": dict(enumerate(train_classes(names))),
        "label_mapping": label_mapping(names),
    }
    path = out / "data.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    counts = {s: sum(v == s for v in splits.values()) for s in SPLITS}
    log.info("wrote %s: %s", path, counts)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw", type=Path, required=True, help="extracted Semantic Drone Dataset")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(prepare(args.raw, args.out, seed=args.seed))


if __name__ == "__main__":
    main()
