"""The visual detector keeps people only (no gear or vehicle boxes), and the person features
ignore any other COCO label."""

import numpy as np
import pytest

from cutpoint.events import visual

BICYCLE, SNOWBOARD = 2, 36


def test_people_frames_count_only_person_boxes():
    # rows: frame, label, score, x1, y1, x2, y2 - a person and a bicycle/snowboard in frame 0
    det = np.array(
        [
            [0, visual.COCO_PERSON, 0.9, 0.1, 0.1, 0.3, 0.5],
            [0, BICYCLE, 0.95, 0.4, 0.1, 0.9, 0.6],
            [0, SNOWBOARD, 0.95, 0.4, 0.2, 0.8, 0.3],
            [1, BICYCLE, 0.99, 0.1, 0.1, 0.9, 0.9],
        ]
    )
    data = visual.VisualData(t=np.array([0.0, 1.0]), emb=np.zeros((2, 3, 768), np.float16), det=det)
    out = visual.people_frames(data)
    assert out["count"].tolist() == [1.0, 0.0]
    assert out["area"][1] == 0.0 and out["close"][1] == 0.0


def test_detector_run_drops_non_person_detections():
    torch = pytest.importorskip("torch")
    boxes = torch.tensor([[10.0, 10, 50, 90], [0, 0, 90, 90], [5, 5, 20, 20]])
    labels = torch.tensor([visual.COCO_PERSON, BICYCLE, SNOWBOARD])
    scores = torch.tensor([0.9, 0.99, 0.99])

    models = object.__new__(visual.VisualModels)
    models.torch, models.device, models.dtype = torch, "cpu", torch.float32
    models.box_min_score, models.box_max = 0.5, 6
    models._embed = lambda c: torch.ones(c.shape[0], 768)  # no CLIP weights in a unit test
    models.det = lambda _xs: [{"boxes": boxes, "labels": labels, "scores": scores}]

    frames = np.zeros((1, 100, 100, 3), np.uint8)
    _, dets, box_embs = models.run(frames)

    assert dets[0][:, 0].tolist() == [float(visual.COCO_PERSON)]
    assert box_embs[0].shape[0] == 1 and not np.isnan(box_embs[0]).any()
