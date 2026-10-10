import math

import numpy as np
import pytest
import yaml
from PIL import Image

from landing.classes import category_luts, label_mapping, train_classes
from landing.data import block_split, decode_rgb, prepare
from landing.evaluate import binary_iou, confusion, iou, load_reference
from landing.zones import LandingConfig, Site, find_site, ground_sampling_distance, site_is_safe

CLASSES = ["paved-area", "grass", "tree", "roof", "person", "water"]
LANDABLE, LIVING = category_luts(CLASSES)
GSD = 0.1  # 10 cm per pixel
CFG = LandingConfig(clearance_m=1.0, living_buffer_m=3.0, margin_m=0.0)


def lawn(h=100, w=100):
    return np.full((h, w), CLASSES.index("grass"), dtype=np.int64)


def test_ground_sampling_distance():
    assert ground_sampling_distance(20, 90, 1000) == pytest.approx(0.04)


def test_open_lawn_lands_straight_below():
    site, clearance = find_site(lawn(), LANDABLE, LIVING, GSD, CFG)
    assert site.found and site.distance_m < GSD
    assert clearance.max() == pytest.approx(5.0, abs=GSD)  # 50 px to the image border


def test_person_below_moves_the_site_out_of_the_buffer():
    m = lawn()
    m[48:52, 48:52] = CLASSES.index("person")
    site, _ = find_site(m, LANDABLE, LIVING, GSD, CFG)
    assert site.found
    assert 4.0 < site.distance_m < 4.5  # buffer + clearance from the person: the shortest diversion
    assert site_is_safe(site, m, LANDABLE, LIVING, GSD, CFG)
    straight_down = Site(True, 49.5, 49.5, 0.0, 0.0)
    assert not site_is_safe(straight_down, m, LANDABLE, LIVING, GSD, CFG)


def test_narrow_path_is_not_a_landing_site():
    m = np.full((100, 100), CLASSES.index("tree"), dtype=np.int64)
    m[:, 45:55] = CLASSES.index("paved-area")  # a 1 m wide path, needs 2 m
    site, _ = find_site(m, LANDABLE, LIVING, GSD, CFG)
    assert not site.found and math.isnan(site.x)


def test_site_needs_clearance_from_obstacles():
    m = np.full((100, 100), CLASSES.index("roof"), dtype=np.int64)
    m[60:90, 10:40] = CLASSES.index("grass")  # a 3 m x 3 m lawn in the corner
    site, clearance = find_site(m, LANDABLE, LIVING, GSD, CFG)
    assert site.found and clearance[round(site.y), round(site.x)] >= 1.0
    assert site_is_safe(site, m, LANDABLE, LIVING, GSD, CFG)


def test_unlabelled_reference_pixels_are_unsafe():
    m = lawn()
    m[45:55, 45:55] = -1
    assert not site_is_safe(Site(True, 50, 50, 1, 0), m, LANDABLE, LIVING, GSD, CFG)


def test_label_mapping_skips_unlabelled_and_conflicting():
    names = ["unlabeled", "paved-area", "person", "conflicting"]
    assert train_classes(names) == ["paved-area", "person"]
    assert label_mapping(names) == {0: "ignore_label", 1: 0, 2: 1, 3: "ignore_label"}


def test_decode_rgb_with_unknown_colour():
    colours = np.array([[0, 0, 0], [128, 64, 128], [255, 22, 96]], dtype=np.uint8)
    mask = np.array([[[128, 64, 128], [255, 22, 96], [1, 2, 3]]], dtype=np.uint8)
    assert decode_rgb(mask, colours).tolist() == [[1, 2, 0]]


def test_block_split_keeps_neighbours_together():
    stems = [f"{i:03d}" for i in range(100)]
    split = block_split(stems)
    assert [sum(v == s for v in split.values()) for s in ("train", "val", "test")] == [70, 10, 20]
    for start in range(0, 100, 10):
        assert len({split[s] for s in stems[start : start + 10]}) == 1


def test_iou_helpers():
    ref = np.array([[0, 1], [1, -1]])
    pred = np.array([[0, 1], [0, 1]])
    cm = confusion(pred, ref, 2)
    assert cm.tolist() == [[1, 0], [1, 1]]
    assert iou(cm).tolist() == [0.5, 0.5]
    assert binary_iou(pred == 1, ref == 1, ref >= 0) == (1, 2)


def test_prepare_writes_ultralytics_layout(tmp_path):
    raw = tmp_path / "raw"
    (raw / "images").mkdir(parents=True)
    (raw / "gt" / "label_images").mkdir(parents=True)
    (raw / "class_dict.csv").write_text(
        "name, r, g, b\nunlabeled, 0, 0, 0\npaved-area, 128, 64, 128\nperson, 255, 22, 96\n"
    )
    for i in range(20):
        Image.new("RGB", (60, 40), (90, 90, 90)).save(raw / "images" / f"{i:03d}.jpg")
        mask = np.zeros((40, 60, 3), np.uint8)
        mask[:, :30] = (128, 64, 128)
        mask[:5, :5] = (255, 22, 96)
        Image.fromarray(mask).save(raw / "gt" / "label_images" / f"{i:03d}.png")
    path = prepare(raw, tmp_path / "out", size=(30, 20))
    data = yaml.safe_load(path.read_text())
    assert data["names"] == {0: "paved-area", 1: "person"}
    assert data["label_mapping"] == {0: "ignore_label", 1: 0, 2: 1}
    test = sorted((tmp_path / "out" / "masks" / "test").glob("*.png"))
    assert test
    ref = load_reference(test[0], data["label_mapping"])
    assert ref.shape == (20, 30) and set(np.unique(ref)) == {-1, 0, 1}


@pytest.mark.parametrize("seed", range(20))
def test_chosen_site_is_safe_on_its_own_map(seed):
    rng = np.random.default_rng(seed)
    m = lawn(120, 160)
    for _ in range(rng.integers(1, 6)):  # random obstacles and people
        y, x = rng.integers(0, 120), rng.integers(0, 160)
        h, w = rng.integers(2, 30, size=2)
        m[y : y + h, x : x + w] = rng.choice([CLASSES.index(c) for c in ("tree", "roof", "person", "water")])
    site, _ = find_site(m, LANDABLE, LIVING, GSD, CFG)
    if site.found:
        assert site_is_safe(site, m, LANDABLE, LIVING, GSD, CFG)


def test_margin_absorbs_a_boundary_error():
    """The plan is made on a map whose person mask is a few pixels too small; the margin keeps the
    chosen site safe on the true map."""
    truth = lawn(100, 160)
    truth[45:55, 75:85] = CLASSES.index("person")
    predicted = lawn(100, 160)
    predicted[48:52, 78:82] = CLASSES.index("person")  # 30 cm short on every side
    tight, _ = find_site(predicted, LANDABLE, LIVING, GSD, CFG)
    assert tight.found and not site_is_safe(tight, truth, LANDABLE, LIVING, GSD, CFG)
    planned = LandingConfig(clearance_m=1.0, living_buffer_m=3.0, margin_m=0.5)
    margin, _ = find_site(predicted, LANDABLE, LIVING, GSD, planned)
    assert margin.found and site_is_safe(margin, truth, LANDABLE, LIVING, GSD, CFG)
