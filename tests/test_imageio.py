"""Pictures written and read through bytes (``cutpoint.imageio``), on a non-ASCII path (#797)."""

from __future__ import annotations

import numpy as np
import pytest

from cutpoint import imageio


@pytest.mark.parametrize(
    ("name", "write"),
    [
        ("x.png", lambda p, img: imageio.write_png(p, img)),
        ("x.jpg", lambda p, img: imageio.write_jpeg(p, img, 90)),
    ],
)
def test_pictures_are_written_to_and_read_from_a_non_ascii_path(tmp_path, name, write):
    folder = tmp_path / "ñame_日本"
    folder.mkdir()
    img = np.full((20, 30, 3), 120, np.uint8)
    assert write(folder / name, img)
    back = imageio.read_image(folder / name)
    assert back is not None and back.shape == img.shape
    assert abs(int(back.mean()) - 120) <= 2
    assert imageio.read_image(folder / "missing.png") is None
