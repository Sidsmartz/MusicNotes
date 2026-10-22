import librosa
import numpy as np
import pytest

from musicnotes import audio
from tests import synth


def _analyze(sig):
    y = librosa.resample(sig, orig_sr=synth.SR, target_sr=audio.SR)
    return audio.analyze(y)


@pytest.fixture(scope="module")
def guitar():
    sig, truth = synth.guitar()
    return _analyze(sig), truth


@pytest.fixture(scope="module")
def piano():
    sig, truth = synth.piano()
    return _analyze(sig), truth


@pytest.fixture(scope="module")
def drums():
    sig, truth = synth.drums()
    return _analyze(sig), truth


def _chord_accuracy(ev, truth, bar_len):
    right = 0
    for t, name in truth:
        mid = t + bar_len / 2
        c = next((c for c in ev.chords if c.t0 <= mid < c.t1), None)
        right += bool(c and c.label.name == name)
    return right / len(truth)


def test_guitar_is_recognised(guitar):
    A, truth = guitar
    assert A.guess.mode == "guitar", A.guess
    ev = audio.build_events(A, "guitar")
    assert _chord_accuracy(ev, truth, 4 * 60 / 96) >= 0.85, [(c.label.name, round(c.t0, 2)) for c in ev.chords]
    assert len(ev.onsets) >= 28  # 32 strums
    fly = [n for n in ev.notes if n.fly]
    assert 20 <= len(fly) <= 70


def test_piano_is_recognised(piano):
    A, truth = piano
    assert A.guess.mode == "piano", A.guess
    ev = audio.build_events(A, "piano")
    assert _chord_accuracy(ev, truth, 4 * 60 / 100) >= 0.85, [(c.label.name, round(c.t0, 2)) for c in ev.chords]
    # The melody notes should be found with the right pitch.
    melody = {72, 76, 79, 69, 77, 71, 74}
    found = {n.midi for n in ev.notes}
    assert len(melody & found) >= 6


def test_drums_are_classified(drums):
    A, truth = drums
    assert A.guess.mode == "drums", A.guess
    ev = audio.build_events(A, "drums")
    for kind in ("kick", "snare", "hat"):
        masked = lambda t: kind == "hat" and any(k == "snare" and abs(u - t) < 0.01 for u, k in truth)
        want = [t for t, k in truth if k == kind and not masked(t)]
        got = [h.t for h in ev.drums if h.kind == kind]
        hit = sum(any(abs(g - w) < 0.05 for g in got) for w in want)
        false = sum(not any(abs(g - w) < 0.05 for w, k in truth if k == kind) for g in got)
        assert hit / len(want) >= 0.8, (kind, hit, len(want))
        assert false / max(1, len(got)) <= 0.25, (kind, false, len(got))
