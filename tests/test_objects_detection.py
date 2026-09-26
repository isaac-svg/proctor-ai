import os
from pathlib import Path

import pytest

import obd
from observations import BoundingBox, ObjectObservation


def det(label, conf, x, y, w, h, source="main"):
    return ObjectObservation(label, conf, BoundingBox(x, y, w, h), source)


# ---- pure helpers ------------------------------------------------------------------------------------

def test_overlapping_boxes_of_one_label_collapse_to_the_most_confident():
    merged = obd.merge_detections([det("cell phone", 0.4, 0.5, 0.5, 0.1, 0.1), det("cell phone", 0.7, 0.51, 0.5, 0.1, 0.1, "hands")])
    assert [(m.confidence, m.source) for m in merged] == [(0.7, "hands")]


def test_different_labels_or_separate_places_are_all_kept():
    merged = obd.merge_detections([det("cell phone", 0.6, 0.1, 0.1, 0.1, 0.1), det("book", 0.6, 0.1, 0.1, 0.1, 0.1), det("cell phone", 0.5, 0.7, 0.7, 0.1, 0.1)])
    assert len(merged) == 3


def test_a_box_from_the_crop_maps_back_to_the_full_frame():
    box = obd.remap_crop_box(0.25, 0.5, 0.75, 1.0, top=0.35)
    assert box.x == 0.25 and box.width == 0.5
    assert box.y == pytest.approx(0.35 + 0.5 * 0.65) and box.height == pytest.approx(0.5 * 0.65)
    assert box.y + box.height == pytest.approx(1.0)  # the bottom of the crop is the bottom of the frame


def test_implausible_sizes_are_dropped():
    assert obd.plausible(det("cell phone", 0.6, 0.4, 0.4, 0.08, 0.12))
    assert not obd.plausible(det("cell phone", 0.9, 0.0, 0.0, 0.9, 0.9))  # a monitor, not a phone
    assert not obd.plausible(det("cell phone", 0.9, 0.4, 0.4, 0.01, 0.01))  # a speck
    assert not obd.plausible(det("cell phone", 0.9, 0.4, 0.4, 0.0, 0.1))
    assert obd.plausible(det("person", 0.9, 0.0, 0.0, 1.0, 1.0))  # someone sitting very close is still a person


def test_settings_come_from_the_environment_and_are_bounded(monkeypatch):
    monkeypatch.setenv("PROCTOR_YOLO_IMGSZ", "99999")
    monkeypatch.setenv("PROCTOR_OBJECT_SECOND_PASS", "off")
    monkeypatch.setenv("PROCTOR_EXTRA_LABELS", '{"Earbud": "earbuds", "watch": "smartwatch"}')
    s = obd.settings()
    assert s["imgsz"] == 1920 and s["second_pass"] is False
    assert s["extra_labels"] == {"earbud": "earbuds", "watch": "smartwatch"}
    monkeypatch.setenv("PROCTOR_EXTRA_LABELS", "not json")
    assert obd.settings()["extra_labels"] == {}
    assert set(obd.describe()) == {"model", "imgsz", "second_pass", "extra_model", "extra_labels"}


# ---- the real model ----------------------------------------------------------------------------------

def _bus():
    ultralytics = pytest.importorskip("ultralytics")
    cv2 = pytest.importorskip("cv2")
    image = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    if not image.exists() or not (Path(obd.__file__).parent / "yolo26n.pt").exists():
        pytest.skip("sample image or model not available")
    return cv2.imread(str(image))


def test_the_real_model_finds_the_people_in_a_known_photo_with_sane_boxes():
    img = _bus()
    found = obd.detect_objects(img)
    people = [o for o in found if o.label == "person"]
    assert len(people) >= 3  # the photo has four; a webcam only needs "more than one"
    assert all(0 <= o.box.x <= 1 and 0 <= o.box.y <= 1 and o.box.x + o.box.width <= 1.01 and o.box.y + o.box.height <= 1.01 for o in found)
    assert all(o.label in ("person", "cell phone", "book", "laptop") for o in found)  # nothing else leaks through


def test_the_second_pass_does_not_lose_anything_the_first_found(monkeypatch):
    img = _bus()
    monkeypatch.setenv("PROCTOR_OBJECT_SECOND_PASS", "0")
    single = [o for o in obd.detect_objects(img) if o.label == "person"]
    monkeypatch.setenv("PROCTOR_OBJECT_SECOND_PASS", "1")
    both = [o for o in obd.detect_objects(img) if o.label == "person"]
    assert len(both) >= len(single)
    assert {o.source for o in both} <= {"main", "hands"}


def test_a_missing_extra_model_does_not_stop_the_normal_detections(monkeypatch):
    img = _bus()
    monkeypatch.setenv("PROCTOR_EXTRA_MODEL", "/nonexistent/model.pt")
    monkeypatch.setenv("PROCTOR_EXTRA_LABELS", '{"earbud": "earbuds"}')
    assert any(o.label == "person" for o in obd.detect_objects(img))
