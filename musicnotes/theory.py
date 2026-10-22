"""Note names, colours, chord templates and guitar chord shapes."""
from __future__ import annotations

import colorsys
from dataclasses import dataclass

import numpy as np

NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
SOLFEGE = ["Do", "Do#", "Re", "Mib", "Mi", "Fa", "Fa#", "Sol", "Lab", "La", "Sib", "Si"]
OPEN_STRINGS = [40, 45, 50, 55, 59, 64]  # E2 A2 D3 G3 B3 E4, low string first


def midi_to_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69) / 12)


def pitch_class(m: int) -> int:
    return int(m) % 12


def note_label(m: int, style: str = "letter") -> str:
    if style == "solfege":
        return SOLFEGE[pitch_class(m)]
    if style == "octave":
        return f"{NAMES[pitch_class(m)]}{m // 12 - 1}"
    return NAMES[pitch_class(m)]


def note_rgb(m: int, lightness: float = 0.6, saturation: float = 0.85) -> tuple[float, float, float]:
    """Colour for a pitch class. Hue follows the circle of fifths, so notes that
    sound good together get neighbouring colours."""
    hue = ((pitch_class(m) * 7) % 12) / 12
    return colorsys.hls_to_rgb(hue, lightness, saturation)


@dataclass(frozen=True)
class ChordType:
    suffix: str
    quality: str
    intervals: tuple[int, ...]
    weight: float


CHORD_TYPES = [
    ChordType("", "major", (0, 4, 7), 1.0),
    ChordType("m", "minor", (0, 3, 7), 1.0),
    ChordType("7", "dominant 7th", (0, 4, 7, 10), 0.96),
    ChordType("m7", "minor 7th", (0, 3, 7, 10), 0.95),
    ChordType("maj7", "major 7th", (0, 4, 7, 11), 0.94),
]


@dataclass(frozen=True)
class ChordLabel:
    name: str
    root: int
    quality: str
    intervals: tuple[int, ...]

    @property
    def tones(self) -> set[int]:
        return {(self.root + i) % 12 for i in self.intervals}


def chord_vocabulary() -> list[ChordLabel]:
    return [ChordLabel(NAMES[r] + t.suffix, r, t.quality, t.intervals) for t in CHORD_TYPES for r in range(12)]


def chord_templates(vocab: list[ChordLabel]) -> np.ndarray:
    """Unit-norm 12-d templates. Each chord tone also carries its octave and a
    little of its fifth, which is what a CQT chroma of a real instrument shows."""
    weight = {ct.quality: ct.weight for ct in CHORD_TYPES}
    t = np.zeros((len(vocab), 12))
    for i, c in enumerate(vocab):
        for iv in c.intervals:
            t[i, (c.root + iv) % 12] += 1.0
            t[i, (c.root + iv + 7) % 12] += 0.15
        t[i] /= np.linalg.norm(t[i])
        t[i] *= weight[c.quality]
    return t


# Open and barre shapes, low E string first. -1 = muted, 0 = open.
GUITAR_SHAPES: dict[str, list[int]] = {
    "C": [-1, 3, 2, 0, 1, 0], "D": [-1, -1, 0, 2, 3, 2], "E": [0, 2, 2, 1, 0, 0], "F": [1, 3, 3, 2, 1, 1],
    "G": [3, 2, 0, 0, 0, 3], "A": [-1, 0, 2, 2, 2, 0], "B": [-1, 2, 4, 4, 4, 2], "Bb": [-1, 1, 3, 3, 3, 1],
    "Eb": [-1, 6, 5, 3, 4, 3], "Ab": [4, 6, 6, 5, 4, 4], "F#": [2, 4, 4, 3, 2, 2], "C#": [-1, 4, 6, 6, 6, 4],
    "Cm": [-1, 3, 5, 5, 4, 3], "Dm": [-1, -1, 0, 2, 3, 1], "Em": [0, 2, 2, 0, 0, 0], "Fm": [1, 3, 3, 1, 1, 1],
    "Gm": [3, 5, 5, 3, 3, 3], "Am": [-1, 0, 2, 2, 1, 0], "Bm": [-1, 2, 4, 4, 3, 2], "F#m": [2, 4, 4, 2, 2, 2],
    "C#m": [-1, 4, 6, 6, 5, 4], "Bbm": [-1, 1, 3, 3, 2, 1], "Ebm": [-1, -1, 1, 3, 4, 2], "Abm": [4, 6, 6, 4, 4, 4],
    "C7": [-1, 3, 2, 3, 1, 0], "D7": [-1, -1, 0, 2, 1, 2], "E7": [0, 2, 0, 1, 0, 0], "F7": [1, 3, 1, 2, 1, 1],
    "G7": [3, 2, 0, 0, 0, 1], "A7": [-1, 0, 2, 0, 2, 0], "B7": [-1, 2, 1, 2, 0, 2], "Bb7": [-1, 1, 3, 1, 3, 1],
    "Am7": [-1, 0, 2, 0, 1, 0], "Em7": [0, 2, 2, 0, 3, 0], "Dm7": [-1, -1, 0, 2, 1, 1], "Bm7": [-1, 2, 0, 2, 0, 2],
    "F#m7": [2, 4, 2, 2, 2, 2], "Cmaj7": [-1, 3, 2, 0, 0, 0], "Dmaj7": [-1, -1, 0, 2, 2, 2],
    "Fmaj7": [-1, -1, 3, 2, 1, 0], "Gmaj7": [3, 2, 0, 0, 0, 2], "Amaj7": [-1, 0, 2, 1, 2, 0], "Emaj7": [0, 2, 1, 1, 0, 0],
}


def guitar_shape(chord: ChordLabel) -> list[int] | None:
    """Standard shape for a chord, generated as a barre shape if not in the table."""
    if chord.name in GUITAR_SHAPES:
        return GUITAR_SHAPES[chord.name]
    # E-shape barre (root on low E) or A-shape barre (root on A), whichever is lower.
    e_root = (chord.root - 4) % 12 or 12
    a_root = (chord.root - 9) % 12 or 12
    e_shapes = {"major": [0, 2, 2, 1, 0, 0], "minor": [0, 2, 2, 0, 0, 0], "dominant 7th": [0, 2, 0, 1, 0, 0],
                "minor 7th": [0, 2, 0, 0, 0, 0], "major 7th": [0, 2, 1, 1, 0, 0]}
    a_shapes = {"major": [-1, 0, 2, 2, 2, 0], "minor": [-1, 0, 2, 2, 1, 0], "dominant 7th": [-1, 0, 2, 0, 2, 0],
                "minor 7th": [-1, 0, 2, 0, 1, 0], "major 7th": [-1, 0, 2, 1, 2, 0]}
    base, shapes = (e_root, e_shapes) if e_root <= a_root else (a_root, a_shapes)
    shape = shapes.get(chord.quality)
    if shape is None:
        return None
    return [f if f < 0 else f + base for f in shape]


def place_on_guitar(midi: int, near_fret: float, shape: list[int] | None, max_fret: int = 15) -> tuple[int, int] | None:
    """String and fret for a note: inside the chord shape if it is there, otherwise
    the playable position closest to where the hand was."""
    if shape:
        for s in range(5, -1, -1):
            if shape[s] >= 0 and OPEN_STRINGS[s] + shape[s] == midi:
                return s, shape[s]
    best = None
    for s in range(6):
        f = midi - OPEN_STRINGS[s]
        if 0 <= f <= max_fret:
            cost = abs(f - near_fret) - (0.5 if f == 0 else 0) + s * 0.05
            if best is None or cost < best[0]:
                best = (cost, s, f)
    return (best[1], best[2]) if best else None
