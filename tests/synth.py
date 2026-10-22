"""Synthesised test clips with known content: Karplus-Strong guitar strums,
an additive piano, and a drum groove."""
from __future__ import annotations

import numpy as np

from musicnotes import theory

SR = 44100


def _pluck(out, t, freq, amp, rng, decay=0.9965):
    start, period = int(t * SR), max(2, int(round(SR / freq)))
    noise = rng.uniform(-1, 1, period)
    buf = (noise + np.roll(noise, 1)) * 0.5
    n = min(len(out) - start, SR * 3)
    res = np.empty(n)
    idx = 0
    for i in range(n):  # Karplus-Strong delay line
        cur = buf[idx]
        nxt = buf[(idx + 1) % period]
        res[i] = cur
        buf[idx] = (cur + nxt) * 0.5 * decay
        idx = (idx + 1) % period
    out[start:start + n] += res * amp


def guitar(progression=("A", "D", "Em", "G"), bpm=96, repeats=2):
    beat = 60 / bpm
    dur = len(progression) * 4 * beat * repeats + 2
    out = np.zeros(int(dur * SR))
    rng = np.random.default_rng(7)
    truth, t = [], 0.4
    for _ in range(repeats):
        for name in progression:
            shape = theory.GUITAR_SHAPES[name]
            truth.append((t, name))
            for b in range(4):
                strings = [s for s in range(6) if shape[s] >= 0]
                if b % 2:
                    strings.reverse()
                for i, s in enumerate(strings):
                    _pluck(out, t + i * 0.012, theory.midi_to_hz(theory.OPEN_STRINGS[s] + shape[s]), 0.32 if b % 2 == 0 else 0.22, rng)
                t += beat
    return _norm(out), truth


def _piano_note(out, t, m, dur, amp):
    f0, start = theory.midi_to_hz(m), int(t * SR)
    n = min(len(out) - start, int((dur + 0.6) * SR))
    tt = np.arange(n) / SR
    for h in range(1, 9):
        f = f0 * h * np.sqrt(1 + 0.0004 * h * h)
        if f > 9000:
            break
        env = np.exp(-tt * (1.2 + h * 0.6)) * np.minimum(1, tt / 0.004)
        env = np.where(tt > dur, env * np.exp(-(tt - dur) * 12), env)
        out[start:start + n] += amp / h ** 1.1 * env * np.sin(2 * np.pi * f * tt)


def piano(repeats=2):
    beat = 60 / 100
    chords = [("C", [48, 52, 55, 60]), ("Am", [45, 52, 57, 60]), ("F", [41, 53, 57, 60]), ("G", [43, 55, 59, 62])]
    melody = [[72, 76, 79, 76], [72, 69, 72, 76], [77, 76, 72, 69], [71, 74, 79, 74]]
    out = np.zeros(int((len(chords) * 4 * beat * repeats + 2) * SR))
    truth, t = [], 0.4
    for _ in range(repeats):
        for ci, (name, notes) in enumerate(chords):
            truth.append((t, name))
            for m in notes:
                _piano_note(out, t, m, beat * 4 - 0.05, 0.11)
            for b, m in enumerate(melody[ci]):
                _piano_note(out, t + b * beat, m, beat * 0.9, 0.16)
            t += 4 * beat
    return _norm(out), truth


def drums(bars=8, bpm=110):
    beat = 60 / bpm
    out = np.zeros(int((bars * 4 * beat + 1.5) * SR))
    rng = np.random.default_rng(3)
    truth = []

    def add(t, sig):
        s = int(t * SR)
        n = min(len(sig), len(out) - s)
        out[s:s + n] += sig[:n]

    tk = np.arange(int(0.4 * SR)) / SR
    kick = 0.9 * np.sin(2 * np.pi * np.cumsum(45 + 110 * np.exp(-tk * 28)) / SR) * np.exp(-tk * 9)
    ts = np.arange(int(0.25 * SR)) / SR
    noise = rng.uniform(-1, 1, len(ts))
    lp = np.zeros_like(noise)
    for i in range(1, len(noise)):
        lp[i] = lp[i - 1] + 0.35 * (noise[i] - lp[i - 1])
    snare = (0.5 * lp + 0.2 * noise) * np.exp(-ts * 16) + 0.35 * np.sin(2 * np.pi * 190 * ts) * np.exp(-ts * 22)
    th = np.arange(int(0.2 * SR)) / SR
    hn = np.diff(np.diff(rng.uniform(-1, 1, len(th) + 2)))
    for bar in range(bars):
        for e in range(8):
            t = 0.4 + (bar * 4 + e / 2) * beat
            add(t, 0.16 * hn * np.exp(-th * (14 if e == 7 and bar % 2 else 45)))
            truth.append((t, "hat"))
            if e in (0, 4) or (e == 5 and bar % 2):
                add(t, kick); truth.append((t, "kick"))
            if e in (2, 6):
                add(t, snare); truth.append((t, "snare"))
    return _norm(out), truth


def _norm(x):
    return (x * 0.89 / np.abs(x).max()).astype(np.float32)
