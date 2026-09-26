import cv2
import numpy as np

from evidence import EvidenceBuffer, draw_annotations
from helpers import face, frame, obj_box
from observations import BoundingBox
from pipeline import ProctorPipeline
from pipeline_config import PipelineConfig

CFG = PipelineConfig()


def gray(w=640, h=480):
    return np.full((h, w, 3), 128, dtype=np.uint8)


def test_a_box_is_drawn_on_a_copy_and_the_original_is_untouched():
    src = gray()
    out = draw_annotations(src, [("cell phone", BoundingBox(0.25, 0.25, 0.25, 0.25))])
    assert (src == 128).all()
    assert not (out == 128).all()
    # The left edge of the box, halfway down it, is the marker colour.
    assert tuple(out[int(0.375 * 480), int(0.25 * 640)]) == (40, 90, 255)


def test_boxes_partly_outside_the_frame_or_empty_do_not_break_drawing():
    out = draw_annotations(gray(), [("x", BoundingBox(-0.2, -0.2, 0.5, 0.5)), ("y", BoundingBox(0.5, 0.5, 0.0, 0.3)), ("z", BoundingBox(0.9, 0.9, 0.5, 0.5))])
    assert out.shape == (480, 640, 3)


def test_the_stored_evidence_thumbnail_carries_the_marking():
    buf = EvidenceBuffer(CFG)
    buf.add_frame(1.0, gray(), [("cell phone", BoundingBox(0.3, 0.3, 0.3, 0.3))])
    (_, jpeg), = buf._frames
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == 320
    # JPEG is lossy, so look for "clearly not gray" around where the box edge is.
    edge = img[int(0.45 * img.shape[0]), int(0.3 * 320) - 2:int(0.3 * 320) + 3].astype(int)
    assert (abs(edge[:, 2] - edge[:, 0]) > 60).any()


def test_without_annotations_the_thumbnail_is_unmarked():
    buf = EvidenceBuffer(CFG)
    buf.add_frame(1.0, gray())
    (_, jpeg), = buf._frames
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).astype(int)
    assert abs(img[:, :, 2] - img[:, :, 0]).max() < 12


def test_the_pipeline_offers_the_confirmed_phone_and_the_counted_people():
    p = ProctorPipeline(CFG)
    for t in range(5):
        p.on_frame(frame(float(t), objects=[obj_box("cell phone", 0.7, 0.5, 0.6)]))
    assert [label for label, _ in p.annotations()] == ["cell phone"]

    q = ProctorPipeline(CFG)
    two = [face(cx=0.3), face(cx=0.7)]
    for t in range(6):
        q.on_frame(frame(float(t), faces=two))
    assert q.multiple.person_count == 2
