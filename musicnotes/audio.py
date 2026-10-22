"""Audio analysis: what was played and when.

Classic signal processing throughout:
  * onsets from spectral flux (librosa), beats and tempo from the onset envelope;
  * chords from a CQT chroma of the harmonic part, scored against chord
    templates and decoded with a Viterbi pass so labels do not flicker;
  * notes from harmonic summation with iterative cancellation (up to six at
    once), tracked over time with hysteresis;
  * drum hits from band-wise flux of the percussive part (kick / snare / hi-hat).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d

from . import theory

SR = 22050
HOP = 256
N_FFT = 4096
MIDI_LO, MIDI_HI, HARMONICS = 28, 96, 8
MAX_POLY = 6
MIN_SALIENCE = 2.0


@dataclass
class Note:
    t0: float
    t1: float
    midi: int
    velocity: float
    fly: bool = False  # whether this note gets a flying-note effect
    pos: tuple[int, int] | None = None  # guitar string and fret, when placed on a guitar


@dataclass
class Chord:
    t0: float
    t1: float
    label: theory.ChordLabel
    confidence: float


@dataclass
class DrumHit:
    t: float
    kind: str  # kick | snare | hat
    velocity: float


@dataclass
class AudioGuess:
    mode: str
    reason: str
    stats: dict = field(default_factory=dict)


@dataclass
class AudioAnalysis:
    duration: float
    onsets: np.ndarray
    onset_strength: np.ndarray
    beats: np.ndarray
    tempo: float
    guess: AudioGuess
    # Per-frame material reused when events are rebuilt for another instrument.
    frame_notes: np.ndarray  # (frames, MAX_POLY) midi or -1
    frame_sal: np.ndarray  # (frames, MAX_POLY)
    rms: np.ndarray
    chroma: np.ndarray  # (12, frames at HOP*2)
    band_flux: np.ndarray  # (3, frames) low / mid / high, percussive part
    gate: float
    onset_env: np.ndarray | None = None
    flatness: np.ndarray | None = None

    def frames_to_time(self, f):
        return np.asarray(f) * HOP / SR


@dataclass
class Events:
    mode: str
    onsets: np.ndarray
    beats: np.ndarray
    notes: list[Note]
    chords: list[Chord]
    drums: list[DrumHit]


def load_audio(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float32)


# ---- multi-pitch ------------------------------------------------------------

def _harmonic_windows(n_bins: int, max_harmonics: int = HARMONICS):
    """For each candidate note and harmonic: the FFT bin range and weight."""
    bin_hz = SR / N_FFT
    up = 2 ** (0.35 / 12)
    rows = []
    for m in range(MIDI_LO, MIDI_HI + 1):
        f0 = theory.midi_to_hz(m)
        cand = []
        for h in range(1, max_harmonics + 1):
            f = f0 * h
            lo, hi = int(np.floor(f / up / bin_hz)), int(np.ceil(f * up / bin_hz))
            if f > 6000 or hi >= n_bins:
                break
            cand.append((lo, hi, 0.84 ** (h - 1)))
        rows.append(cand)
    return rows


def _window_max(L: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return L[lo:hi + 1].max(axis=0)


def _salience(L: np.ndarray, windows) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic-sum salience per candidate and frame, plus the share of the
    strongest harmonic found at the fundamental (to reject sub-harmonics)."""
    n = L.shape[1]
    S = np.zeros((len(windows), n), np.float32)
    f0_ratio = np.zeros((len(windows), n), np.float32)
    for i, cand in enumerate(windows):
        peak = np.zeros(n, np.float32)
        for j, (lo, hi, w) in enumerate(cand):
            mx = _window_max(L, lo, hi)
            S[i] += w * mx
            peak = np.maximum(peak, mx)
            if j == 0:
                f0 = mx
        f0_ratio[i] = f0 / np.maximum(peak, 1e-6)
    return S, f0_ratio


def multi_pitch(y: np.ndarray, rms: np.ndarray, gate: float, gain: float):
    win = np.hanning(N_FFT)
    spec = np.abs(librosa.stft(y * gain, n_fft=N_FFT, hop_length=HOP, window=win, center=True))
    spec *= 2 / win.sum()
    k_max = int(6200 / (SR / N_FFT))
    L = np.log1p(1000 * spec[:k_max]).astype(np.float32)
    # Whitening: keep only what stands above the local spectral level, so broadband
    # attack noise does not add up into phantom notes.
    L = np.maximum(0, L - uniform_filter1d(L, size=11, axis=0)).astype(np.float32)
    n = min(L.shape[1], len(rms))
    L = L[:, :n]
    windows = _harmonic_windows(k_max)
    # Cancelling uses every harmonic up to 6 kHz, or the high partials of low
    # strings would come back as phantom high notes.
    cancel = _harmonic_windows(k_max, max_harmonics=64)
    notes = np.full((n, MAX_POLY), -1, np.int16)
    sal = np.zeros((n, MAX_POLY), np.float32)
    active = rms[:n] > gate
    first = np.zeros(n, np.float32)
    work = L.copy()
    for it in range(MAX_POLY):
        S, ratio = _salience(work, windows)
        S[ratio < 0.3] = 0  # no real fundamental: a phantom an octave or fifth below
        best = S.argmax(0)
        val = S[best, np.arange(n)]
        if it == 0:
            first = val
        ok = active & (val >= MIN_SALIENCE) & (val >= 0.3 * first)
        if not ok.any():
            break
        notes[ok, it] = best[ok] + MIDI_LO
        sal[ok, it] = val[ok]
        # Remove the accepted note's harmonics so they don't come back as new notes.
        for c in np.unique(best[ok]):
            frames = np.where(ok & (best == c))[0]
            for lo, hi, _ in cancel[c]:
                work[max(0, lo - 1):hi + 2, frames] *= 0.12
    return notes, sal


# ---- main analysis ------------------------------------------------------------

def analyze(y: np.ndarray, progress=None) -> AudioAnalysis:
    report = progress or (lambda p, msg="": None)
    duration = len(y) / SR
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))
    frame_rms = librosa.feature.rms(y=y, frame_length=2 * HOP, hop_length=HOP, center=True)[0]
    loud_ref = np.percentile(frame_rms, 95) or 1e-6
    gain = 0.2 / loud_ref
    rms = frame_rms * gain
    gate = 0.03
    report(0.1, "Splitting harmonic and percussive sound")
    y_harm, y_perc = librosa.effects.hpss(y, margin=2.0)

    report(0.25, "Finding note onsets and beats")
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP, n_fft=1024, aggregate=np.median, lag=1)
    tempo, beats = librosa.beat.beat_track(onset_envelope=env, sr=SR, hop_length=HOP, units="time")
    tempo = float(np.atleast_1d(tempo)[0])

    report(0.4, "Reading chords")
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=SR, hop_length=HOP * 2, fmin=librosa.note_to_hz("E1"), n_octaves=6)

    report(0.55, "Detecting notes")
    frame_notes, frame_sal = multi_pitch(y, rms, gate, gain)

    report(0.85, "Listening for drums")
    P = np.abs(librosa.stft(y_perc * gain, n_fft=1024, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=1024)
    logP = np.log1p(1000 * P * (2 / 512))
    flux = np.maximum(0, np.diff(logP, axis=1, prepend=logP[:, :1]))
    bands = [(30, 150), (1000, 4000), (6000, 10500)]
    band_flux = np.stack([flux[(freqs >= a) & (freqs <= b)].sum(0) for a, b in bands])

    n = min(len(rms), len(env), frame_notes.shape[0], band_flux.shape[1])
    A = AudioAnalysis(
        duration=duration, onsets=np.array([]), onset_strength=np.array([]), beats=np.asarray(beats), tempo=tempo,
        guess=AudioGuess("melody", ""), frame_notes=frame_notes[:n], frame_sal=frame_sal[:n], rms=rms[:n],
        chroma=chroma, band_flux=band_flux[:, :n], gate=gate,
    )
    A.onset_env = env[:n]
    A.onsets, A.onset_strength = pick_onsets(A, 0.5)
    # Spectral flatness over 100 Hz - 8 kHz: high for noisy drum hits, low for pitched notes.
    F = np.abs(librosa.stft(y * gain, n_fft=2048, hop_length=HOP)) ** 2 + 1e-10
    fband = librosa.fft_frequencies(sr=SR, n_fft=2048)
    F = F[(fband >= 100) & (fband <= 8000)]
    A.flatness = (np.exp(np.log(F).mean(0)) / F.mean(0))[:n]
    A.guess = guess_instrument(A)
    report(1.0, "Done")
    return A


def _peaks(curve: np.ndarray, sensitivity: float, min_gap_s: float = 0.06):
    ref = np.percentile(curve, 99) or 1.0
    c = curve / ref
    delta = 0.28 - 0.22 * sensitivity
    wait = int(min_gap_s * SR / HOP)
    w = int(0.1 * SR / HOP)
    idx = librosa.util.peak_pick(c, pre_max=3, post_max=3, pre_avg=w, post_avg=w, delta=delta, wait=wait)
    return idx, np.clip(c[idx], 0, 1)


def pick_onsets(A: AudioAnalysis, sensitivity: float, curve: np.ndarray | None = None):
    idx, strength = _peaks(A.onset_env if curve is None else curve, sensitivity)
    # Drop "onsets" in silence.
    loud = np.array([A.rms[i:i + 7].max() if i < len(A.rms) else 0 for i in idx])
    keep = loud >= A.gate * 0.3
    return A.frames_to_time(idx[keep]), strength[keep]


# ---- events ---------------------------------------------------------------------

def track_notes(A: AudioAnalysis, max_poly: int):
    """Hysteresis: a note turns on after 3 frames present and off after 6 absent."""
    n = len(A.rms)
    present = np.zeros((MIDI_HI + 1, n), np.float32)
    for k in range(max_poly):
        m, s = A.frame_notes[:, k], A.frame_sal[:, k]
        ok = m >= 0
        present[m[ok], np.where(ok)[0]] = s[ok]
    segs = []
    for m in range(MIDI_LO, MIDI_HI + 1):
        row = present[m]
        if not row.any():
            continue
        on, start, peak, prun, arun = False, 0, 0.0, 0, 0
        for f in range(n):
            if row[f] > 0:
                prun += 1; arun = 0
            else:
                arun += 1; prun = 0
            if not on and prun >= 3:
                on, start, peak = True, f - 2, 0.0
            if on:
                peak = max(peak, row[f])
                if arun >= 6:
                    on = False
                    if f - 5 - start >= 4:
                        segs.append((start, f - 5, m, peak))
        if on:
            segs.append((start, n - 1, m, peak))
    segs.sort()
    return segs


def chord_segments(A: AudioAnalysis, min_dur: float = 0.3) -> list[Chord]:
    vocab = theory.chord_vocabulary()
    T = theory.chord_templates(vocab)
    C = A.chroma
    energy = C.sum(0)
    Cn = C / np.maximum(np.linalg.norm(C, axis=0, keepdims=True), 1e-9)
    scores = T @ Cn  # (chords, frames)
    # "No chord" state: wins when the music is quiet or the chroma matches nothing well.
    hop_frames = 2  # chroma hop is twice the analysis hop
    rms = A.rms[::hop_frames][: C.shape[1]]
    rms = np.pad(rms, (0, max(0, C.shape[1] - len(rms))))
    none = np.where(rms > A.gate, 0.72, 1.2)[None, :]
    S = np.vstack([scores, none])
    prob = np.exp((S - S.max(0)) * 25)
    prob /= prob.sum(0, keepdims=True)
    path = librosa.sequence.viterbi_discriminative(prob, librosa.sequence.transition_loop(len(S), 0.985))
    t = librosa.frames_to_time(np.arange(C.shape[1] + 1), sr=SR, hop_length=HOP * 2)
    out: list[Chord] = []
    start = 0
    for i in range(1, len(path) + 1):
        if i == len(path) or path[i] != path[start]:
            state = path[start]
            if state < len(vocab) and t[i] - t[start] >= min_dur and energy[start:i].mean() > 0:
                conf = float(scores[state, start:i].mean())
                out.append(Chord(float(t[start]), float(t[i]), vocab[state], conf))
            start = i
    # Merge neighbours with the same label across tiny gaps.
    merged: list[Chord] = []
    for c in out:
        if merged and merged[-1].label.name == c.label.name and c.t0 - merged[-1].t1 < 0.5:
            merged[-1].t1 = c.t1
        else:
            merged.append(c)
    return merged


def drum_hits(A: AudioAnalysis, sensitivity: float) -> tuple[list[DrumHit], np.ndarray]:
    bf = A.band_flux
    refs = np.percentile(bf, 99, axis=1)[:, None] + 1e-9
    curve = (bf / refs).max(0)  # each band has an equal say, so quiet hi-hats count
    times, strength = pick_onsets(A, sensitivity, curve)
    idx = (times * SR / HOP).round().astype(int)
    if not len(idx):
        return [], times
    at = lambda b: np.maximum(bf[b, idx], bf[b, np.minimum(idx + 1, bf.shape[1] - 1)])
    lows, mids, highs = (at(b) for b in range(3))
    norm = lambda v: v / (np.percentile(v, 90) + 1e-9)
    lows, mids, highs = norm(lows), norm(mids), norm(highs)
    hits = []
    for t, s, lo, mi, hi in zip(times, strength, lows, mids, highs):
        kinds = []
        if lo > 0.6 and lo > 0.9 * mi:
            kinds.append("kick")
        if mi > 0.85 and mi > 0.8 * hi:
            kinds.append("snare")
        elif hi > 0.25:
            kinds.append("hat")
        if not kinds:
            kinds.append(["kick", "snare", "hat"][int(np.argmax([lo, mi, hi]))])
        hits += [DrumHit(float(t), k, float(s)) for k in kinds]
    return hits, times


FLY_LIMIT = {"piano": 4, "guitar": 2, "melody": 1}


def build_events(A: AudioAnalysis, mode: str, sensitivity: float = 0.5) -> Events:
    onsets, strength = pick_onsets(A, sensitivity)
    if mode == "drums":
        hits, drum_onsets = drum_hits(A, sensitivity)
        return Events(mode, drum_onsets, A.beats, [], [], hits)

    max_poly = 1 if mode == "melody" else MAX_POLY
    segs = track_notes(A, max_poly)
    to_t = lambda f: float(f * HOP / SR)
    onset_f = (onsets * SR / HOP).round().astype(int)
    chords = chord_segments(A) if mode != "melody" else []

    def chord_at(t):
        for c in chords:
            if c.t0 <= t + 0.05 < c.t1 + 0.05:
                return c
        return None

    notes: list[Note] = []
    max_sal = max((s[3] for s in segs), default=1.0) or 1.0
    for f0, f1, m, peak in segs:
        # Snap starts to a nearby onset for tight timing.
        if len(onset_f):
            j = np.argmin(np.abs(onset_f - f0))
            if abs(onset_f[j] - f0) <= 5:
                f0 = min(f0, int(onset_f[j]))
        notes.append(Note(to_t(f0), to_t(f1), int(m), float(min(1.0, peak / max_sal))))
    notes.sort(key=lambda n: (n.t0, -n.velocity))

    # Choose which notes fly: the strongest few per strike, and on guitar only
    # chord tones that start on a strum (others are usually overtones).
    limit = FLY_LIMIT.get(mode, 2)
    i = 0
    while i < len(notes):
        j = i
        while j < len(notes) and notes[j].t0 - notes[i].t0 < 0.05:
            j += 1
        group = sorted(notes[i:j], key=lambda n: -n.velocity)
        on_strike = mode != "guitar" or (len(onsets) and np.min(np.abs(onsets - notes[i].t0)) < 0.06)
        chord = chord_at(notes[i].t0) if mode == "guitar" else None
        flying = 0
        for n in group:
            if on_strike and flying < limit and n.velocity > 0.25 and (chord is None or n.midi % 12 in chord.label.tones):
                n.fly = True
                flying += 1
        i = j
    return Events(mode, onsets, A.beats, notes, chords, [])


# ---- instrument guess from sound alone ---------------------------------------------

def guess_instrument(A: AudioAnalysis) -> AudioGuess:
    loud = A.rms > A.gate
    if not loud.any():
        return AudioGuess("melody", "very little sound")
    flat = float(A.flatness[loud[: len(A.flatness)]].mean())
    voiced_frames = (A.frame_notes[loud] >= 0).any(1)
    voiced = float(voiced_frames.mean())
    poly = float((A.frame_notes[loud][voiced_frames] >= 0).sum(1).mean()) if voiced_frames.any() else 0.0
    pitches = A.frame_notes[loud][A.frame_notes[loud] >= 0]
    lo, hi = (np.percentile(pitches, [5, 95]) if len(pitches) else (0, 0))
    idx = (A.onsets * SR / HOP).round().astype(int)
    chordy = float(np.mean([(A.frame_notes[min(i + 4, len(A.frame_notes) - 1)] >= 0).sum() >= 3 for i in idx])) if len(idx) else 0.0
    stats = dict(flatness=round(flat, 3), voiced=round(voiced, 3), polyphony=round(poly, 2), low=int(lo), high=int(hi), chordy=round(chordy, 2))
    name = lambda m: theory.note_label(int(m), "octave")
    if flat > 0.12 or voiced < 0.35 or (flat > 0.06 and voiced < 0.75):
        return AudioGuess("drums", "mostly noisy, unpitched hits", stats)
    if poly < 1.7:
        return AudioGuess("melody", f"mostly one note at a time, {name(lo)}–{name(hi)}", stats)
    if chordy > 0.6 and lo >= 38 and hi - lo <= 34:
        return AudioGuess("guitar", f"strummed chords in guitar range, {name(lo)}–{name(hi)}", stats)
    return AudioGuess("piano", f"chords and melody across {name(lo)}–{name(hi)}", stats)
