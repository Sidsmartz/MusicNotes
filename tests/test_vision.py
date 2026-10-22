import numpy as np
import pytest

from musicnotes.vision.detector import InstrumentDetector, PoseDetector
from musicnotes.vision.guitar import estimate_neck, fret_u
from musicnotes.vision.piano import estimate_keyboard
from tests import drawings


@pytest.mark.parametrize("angle", [-20.0, 0.0, 15.0])
def test_guitar_neck_and_frets(angle):
    img, box, nut, f12, length = drawings.guitar(angle)
    nm = estimate_neck(img, box)
    assert nm is not None
    assert nm.frets_fitted
    tol = 0.025 * length
    assert np.linalg.norm(nm.nut - nut) < tol, (nm.nut, nut)
    assert np.linalg.norm(nm.fret12 - f12) < tol, (nm.fret12, f12)
    # The 5th fret lands where the drawing put it.
    want = nut + (f12 - nut) * fret_u(5)
    assert np.linalg.norm(nm.point(fret_u(5), 0.5) - want) < tol


@pytest.mark.parametrize("tilt", [0.0, 4.0])
def test_keyboard_black_keys(tilt):
    img, box, black_centre, ww = drawings.keyboard(tilt_deg=tilt)
    km = estimate_keyboard(img, box)
    assert km is not None
    # Every drawn black key is predicted at the right place with the right name.
    for octave in range(3):
        for pc in (1, 3, 6, 8, 10):
            truth = black_centre(octave, pc)
            midi = km.c_midi + 12 * octave + pc
            pred = km.point(km.key_w(midi)[0], 0.315)
            assert np.linalg.norm(pred - truth) < 0.3 * ww, (octave, pc, pred, truth)


def test_models_load_and_run():
    det, pose = InstrumentDetector(), PoseDetector()
    assert "Guitar" in det.names and "Piano" in det.names and "Drum" in det.names
    blank = np.full((480, 640, 3), 127, np.uint8)
    assert det(blank) == []
    assert pose(blank) == []
