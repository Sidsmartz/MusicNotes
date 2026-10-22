"""Draws the effects onto video frames.

Everything is a pure function of the playback time and the analysis, so any
frame can be rendered on its own. Vector shapes use Cairo (anti-aliased);
text uses Pillow with the bundled Fredoka font.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cairo
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import theory
from .audio import Events
from .vision.guitar import NeckModel, fret_u, string_v
from .vision.video import VisionResult

FONT = Path(__file__).resolve().parent / "assets" / "fonts" / "Fredoka.ttf"
TAU = 2 * math.pi
INK = (0.11, 0.13, 0.19)
ACCENT = (0.96, 0.65, 0.14)
DRUM_COLORS = {"kick": (1.0, 0.42, 0.34), "snare": (1.0, 0.79, 0.24), "hat": (0.37, 0.84, 0.91)}
DRUM_WORDS = {"kick": ["BOOM", "DOOM", "BUM!"], "snare": ["PAH!", "CRACK", "TAK!"], "hat": ["tss", "tik", "ts"]}


@dataclass
class RenderOptions:
    notes: bool = True  # flying note characters
    tutorial: bool = True  # chord shapes on frets, lit keys, drum flashes
    outline: str = "neon"  # neon | dashed | ""
    chord_card: bool = True
    timeline: bool = True  # upcoming chords along the bottom
    labels: str = "letter"  # letter | octave | solfege | ""
    cute: bool = True  # faces and sparkles
    shake: bool = True  # kick drum shake


def _hash(i: int, salt: int) -> float:
    x = math.sin(i * 127.1 + salt * 311.7) * 43758.5453
    return x - math.floor(x)


def _ease_out_back(x: float) -> float:
    c1 = 1.70158
    return 1 + (c1 + 1) * (x - 1) ** 3 + c1 * (x - 1) ** 2


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


# ---- text --------------------------------------------------------------------

@lru_cache(maxsize=64)
def _font(size: int, weight: str) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(FONT), size)
    f.set_variation_by_name(weight)
    return f


@lru_cache(maxsize=2048)
def _text_surface(text: str, size: int, rgb: tuple, weight: str, stroke: int, stroke_rgb: tuple) -> cairo.ImageSurface:
    font = _font(size, weight)
    l, t, r, b = font.getbbox(text, stroke_width=stroke)
    w, h = max(1, r - l + 4), max(1, b - t + 4)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((2 - l, 2 - t), text, font=font, fill=tuple(int(c * 255) for c in rgb) + (255,),
                             stroke_width=stroke, stroke_fill=tuple(int(c * 255) for c in stroke_rgb) + (255,))
    a = np.asarray(img).astype(np.float32)
    alpha = a[:, :, 3:4] / 255.0
    bgra = np.empty((h, w, 4), np.uint8)
    bgra[:, :, :3] = (a[:, :, [2, 1, 0]] * alpha).astype(np.uint8)  # premultiplied
    bgra[:, :, 3] = a[:, :, 3].astype(np.uint8)
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    stride = surf.get_stride()
    buf = np.ndarray((h, stride // 4, 4), np.uint8, buffer=surf.get_data())
    buf[:, :w] = bgra
    surf.mark_dirty()
    return surf


def text(ctx, s: str, x: float, y: float, size: float, rgb=INK, weight="Bold", anchor="mm", stroke=0, stroke_rgb=(1, 1, 1), alpha=1.0):
    surf = _text_surface(s, max(6, int(round(size))), tuple(rgb), weight, int(round(stroke)), tuple(stroke_rgb))
    w, h = surf.get_width(), surf.get_height()
    dx = {"l": 0, "m": -w / 2, "r": -w}[anchor[0]]
    dy = {"t": 0, "m": -h / 2, "b": -h}[anchor[1]]
    ctx.save()
    ctx.set_source_surface(surf, x + dx, y + dy)
    ctx.paint_with_alpha(alpha)
    ctx.restore()
    return w, h


def text_width(s: str, size: float, weight="Bold") -> float:
    font = _font(max(6, int(round(size))), weight)
    l, _, r, _ = font.getbbox(s)
    return r - l


# ---- shapes ------------------------------------------------------------------

def rounded_rect(ctx, x, y, w, h, r):
    r = min(r, w / 2, h / 2)
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -TAU / 4, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, TAU / 4)
    ctx.arc(x + r, y + h - r, r, TAU / 4, TAU / 2)
    ctx.arc(x + r, y + r, r, TAU / 2, 3 * TAU / 4)
    ctx.close_path()


def poly(ctx, pts):
    ctx.new_path()
    for i, (x, y) in enumerate(pts):
        (ctx.line_to if i else ctx.move_to)(float(x), float(y))
    ctx.close_path()


def glow_stroke(ctx, path, rgb, width, pulse=0.0, style="neon", t=0.0):
    """Neon tube (soft glow plus a bright core) or marching dashes."""
    ctx.save()
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    if style == "dashed":
        ctx.set_dash([width * 4, width * 3], -t * width * 14)
        ctx.set_source_rgba(0, 0, 0, 0.45)
        ctx.set_line_width(width * 2.2)
        path(); ctx.stroke()
        ctx.set_source_rgba(*rgb, 1)
        ctx.set_line_width(width * (1.2 + 0.5 * pulse))
        path(); ctx.stroke()
    else:
        for mul, a in ((7, 0.06), (4.5, 0.12), (2.6, 0.24)):
            ctx.set_source_rgba(*rgb, a * (1 + pulse))
            ctx.set_line_width(width * mul * (1 + 0.35 * pulse))
            path(); ctx.stroke()
        ctx.set_source_rgba(*rgb, 1)
        ctx.set_line_width(width * 1.3)
        path(); ctx.stroke()
        ctx.set_source_rgba(1, 1, 1, 0.85)
        ctx.set_line_width(width * 0.45)
        path(); ctx.stroke()
    ctx.restore()


def star(ctx, x, y, r, rot):
    ctx.new_path()
    for i in range(8):
        a = rot + i * math.pi / 4
        rr = r if i % 2 == 0 else r * 0.38
        ctx.line_to(x + math.cos(a) * rr, y + math.sin(a) * rr)
    ctx.close_path()
    ctx.fill()


def heart(ctx, x, y, s):
    ctx.new_path()
    ctx.move_to(x, y + s * 0.35)
    ctx.curve_to(x - s * 1.1, y - s * 0.35, x - s * 0.45, y - s * 1.05, x, y - s * 0.45)
    ctx.curve_to(x + s * 0.45, y - s * 1.05, x + s * 1.1, y - s * 0.35, x, y + s * 0.35)
    ctx.fill()


def face(ctx, x, y, s, surprised=False):
    ctx.set_source_rgb(*INK)
    for ex in (-0.32, 0.32):
        ctx.save(); ctx.translate(x + ex * s, y); ctx.scale(0.13 * s, 0.16 * s); ctx.arc(0, 0, 1, 0, TAU); ctx.restore(); ctx.fill()
    ctx.set_source_rgb(1, 1, 1)
    for ex in (-0.32, 0.32):
        ctx.arc(x + ex * s + s * 0.04, y - s * 0.06, s * 0.045, 0, TAU); ctx.fill()
    ctx.set_source_rgb(*INK)
    ctx.set_line_width(s * 0.08)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.new_path()
    if surprised:
        ctx.arc(x, y + s * 0.3, s * 0.11, 0, TAU)
    else:
        ctx.arc(x, y + s * 0.16, s * 0.16, 0.15 * math.pi, 0.85 * math.pi)
    ctx.stroke()
    ctx.set_source_rgba(1, 0.47, 0.59, 0.55)
    for ex in (-0.55, 0.55):
        ctx.save(); ctx.translate(x + ex * s, y + s * 0.2); ctx.scale(0.13 * s, 0.08 * s); ctx.arc(0, 0, 1, 0, TAU); ctx.restore(); ctx.fill()


def note_glyph(ctx, s, rgb, flags, cute):
    """An eighth note with a face, origin at the centre of the head."""
    sx, top = s * 0.56, -s * 2.1
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for col, width in (((1, 1, 1), s * 0.36), (INK, s * 0.16)):
        ctx.set_source_rgb(*col)
        ctx.set_line_width(width)
        ctx.move_to(sx, -s * 0.05); ctx.line_to(sx, top); ctx.stroke()
        for f in range(flags):
            y = top + f * s * 0.5
            ctx.move_to(sx, y)
            ctx.curve_to(sx + s * 0.95, y + s * 0.35, sx + s * 0.85, y + s * 0.85, sx + s * 0.45, y + s * 1.25)
            ctx.stroke()
    ctx.save()
    ctx.rotate(-0.36)
    ctx.scale(s * 0.7, s * 0.52)
    ctx.arc(0, 0, 1, 0, TAU)
    ctx.restore()
    ctx.set_source_rgb(*rgb)
    ctx.fill_preserve()
    ctx.set_source_rgb(*INK)
    ctx.set_line_width(s * 0.12)
    ctx.stroke()
    if cute:
        face(ctx, -s * 0.04, -s * 0.02, s * 0.5)


# ---- the overlay -------------------------------------------------------------

class Overlay:
    def __init__(self, size: tuple[int, int], events: Events, vision: VisionResult, opts: RenderOptions):
        self.W, self.H = size
        self.u = min(self.W, self.H) / 720
        self.ev = events
        self.vr = vision
        self.o = opts
        self.mode = events.mode
        self.fly = [n for n in events.notes if n.fly]
        self.fly_t = np.array([n.t0 for n in self.fly])
        self.onsets = np.asarray(events.onsets)
        self.chord_t = np.array([c.t0 for c in events.chords])
        self.drum_t = np.array([d.t for d in events.drums])
        self.kick_t = np.array([d.t for d in events.drums if d.kind == "kick"])
        self._place_guitar_notes()

    # -- helpers --
    def _since_onset(self, t):
        i = np.searchsorted(self.onsets, t + 1e-6) - 1
        return t - self.onsets[i] if i >= 0 else 9.0

    def _chord_at(self, t):
        i = np.searchsorted(self.chord_t, t + 1e-6) - 1
        if i >= 0 and t <= self.ev.chords[i].t1:
            return i, self.ev.chords[i]
        return None, None

    def _place_guitar_notes(self):
        """String and fret for every note (inside the current chord shape when possible)."""
        if self.mode != "guitar":
            return
        near = 3.0
        for n in sorted(self.ev.notes, key=lambda n: n.t0):
            _, c = self._chord_at(n.t0 + 0.02)
            shape = theory.guitar_shape(c.label) if c else None
            p = theory.place_on_guitar(n.midi, near, shape)
            n.pos = p
            if p and not shape and p[1] > 0:
                near = p[1]

    def neck(self, t) -> NeckModel | None:
        return self.vr.neck_at(t) if self.mode == "guitar" else None

    # -- frame --
    def draw(self, frame: np.ndarray, t: float) -> np.ndarray:
        if self.o.shake and self.mode == "drums":
            dx, dy = self._shake(t)
            if dx or dy:
                M = np.float32([[1.02, 0, dx - 0.01 * self.W], [0, 1.02, dy - 0.01 * self.H]])
                frame = cv2.warpAffine(frame, M, (self.W, self.H), borderMode=cv2.BORDER_REFLECT)
        buf = np.empty((self.H, self.W, 4), np.uint8)
        buf[:, :, :3] = frame
        buf[:, :, 3] = 255
        surf = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_RGB24, self.W, self.H, self.W * 4)
        ctx = cairo.Context(surf)
        ctx.set_antialias(cairo.ANTIALIAS_BEST)
        if self.mode == "guitar":
            self._guitar(ctx, t)
        elif self.mode == "piano":
            self._piano(ctx, t)
        elif self.mode == "drums":
            self._drums(ctx, t)
        else:
            self._melody(ctx, t)
        if self.o.notes and self.mode != "drums":
            self._flying_notes(ctx, t)
        if self.o.chord_card and self.mode in ("guitar", "piano"):
            self._chord_card(ctx, t)
        if self.o.timeline and self.mode in ("guitar", "piano"):
            self._timeline(ctx, t)
        surf.flush()
        return buf[:, :, :3]

    def _shake(self, t):
        i0 = np.searchsorted(self.kick_t, t - 0.3)
        i1 = np.searchsorted(self.kick_t, t + 1e-6)
        x = y = 0.0
        for i in range(i0, i1):
            age = t - self.kick_t[i]
            amp = 6 * self.u * math.exp(-age * 14)
            x += math.sin(age * 70 + i) * amp
            y += math.cos(age * 85 + i) * amp
        return x, y

    # -- instrument outline when there is no detailed geometry --
    def _box_outline(self, ctx, t, rgb=ACCENT):
        box = self.vr.box_at(t)
        if box is None or not self.o.outline:
            return None
        x0, y0, x1, y1 = box
        d = self._since_onset(t)
        pulse = max(0.0, 1 - d / 0.35)
        glow_stroke(ctx, lambda: rounded_rect(ctx, x0, y0, x1 - x0, y1 - y0, 18 * self.u), rgb, 2.2 * self.u, pulse, self.o.outline, t)
        return box

    # -- guitar --
    def _guitar(self, ctx, t):
        nm = self.neck(t)
        if nm is None:
            self._box_outline(ctx, t)
            return
        u = self.u
        P = lambda uu, vv: nm.point(uu, vv)
        d_on = self._since_onset(t)
        pulse = max(0.0, 1 - d_on / 0.35)
        end = fret_u(15)
        width_at = lambda uu: nm.width_nut + (nm.width_12 - nm.width_nut) * uu
        ci, chord = self._chord_at(t)
        shape = theory.guitar_shape(chord.label) if chord else None

        if self.o.outline:
            corners = [P(-0.07, -0.06), P(end, -0.06), P(end, 1.06), P(-0.07, 1.06)]
            glow_stroke(ctx, lambda: poly(ctx, corners), ACCENT, 2.0 * u, pulse, self.o.outline, t)
        if self.o.tutorial:
            ctx.save()
            ctx.set_source_rgba(1, 1, 1, 0.3)
            ctx.set_line_width(1.2 * u)
            for f in range(1, 16):
                a, b = P(fret_u(f), 0.02), P(fret_u(f), 0.98)
                ctx.move_to(*a); ctx.line_to(*b); ctx.stroke()
            ctx.restore()
            # Fret numbers along the low-E edge, like a tutorial fretboard.
            for f in (3, 5, 7, 9, 12):
                p = P((fret_u(f - 1) + fret_u(f)) / 2, -0.32)
                text(ctx, str(f), p[0], p[1], 13 * u, (1, 1, 1), "SemiBold", stroke=3 * u, stroke_rgb=INK, alpha=0.8)

        # Strings shimmer after each strum.
        if self.o.outline and d_on < 1.0:
            amp = math.exp(-d_on * 5) * width_at(0.6) * 0.06
            ctx.save()
            ctx.set_line_cap(cairo.LINE_CAP_ROUND)
            for s in range(6):
                f = shape[s] if shape else 0
                if f < 0:
                    continue
                v = string_v(s)
                u0 = 0.0 if f == 0 else fret_u(f - 1) + 0.62 * (fret_u(f) - fret_u(f - 1))
                rgb = theory.note_rgb(theory.OPEN_STRINGS[s] + max(0, f), 0.66)
                ctx.new_path()
                for k in range(29):
                    uu = u0 + (end + 0.25 - u0) * k / 28
                    base = P(uu, v)
                    nrm = P(uu, v + 0.05) - base
                    nrm /= (np.linalg.norm(nrm) or 1)
                    off = amp * math.sin(math.pi * k / 28) * math.sin(d_on * 70 + s * 1.3)
                    ctx.line_to(*(base + nrm * off))
                for w_, a_ in ((5 * u, 0.18), (2 * u, 0.9)):
                    ctx.set_source_rgba(*rgb, a_ * min(1.0, 1.4 - d_on))
                    ctx.set_line_width(w_)
                    ctx.stroke_preserve()
                ctx.new_path()
            ctx.restore()

        if not self.o.tutorial:
            return

        def dot(s, f, midi, alpha, grow):
            v = string_v(s)
            rgb = theory.note_rgb(midi, 0.58)
            if f > 0:
                hv = 0.075
                ctx.set_source_rgba(*rgb, 0.45 * alpha)
                poly(ctx, [P(fret_u(f - 1), v - hv), P(fret_u(f), v - hv), P(fret_u(f), v + hv), P(fret_u(f - 1), v + hv)])
                ctx.fill()
            c = nm.fret_point(f, s) if f > 0 else P(-0.04, v)
            uu = fret_u(max(f, 1))
            rad = max(9 * u, width_at(uu) * 0.085) * grow
            ctx.save()
            for mul, a in ((2.0, 0.18), (1.45, 0.3)):
                ctx.set_source_rgba(*rgb, a * alpha)
                ctx.arc(*c, rad * mul, 0, TAU); ctx.fill()
            ctx.arc(*c, rad, 0, TAU)
            ctx.set_source_rgba(*(rgb if f > 0 else (1, 1, 1)), alpha)
            ctx.fill_preserve()
            ctx.set_source_rgba(*((1, 1, 1) if f > 0 else rgb), alpha)
            ctx.set_line_width(max(1.5, rad * 0.18))
            ctx.stroke()
            ctx.restore()
            if self.o.labels:
                text(ctx, theory.note_label(midi, self.o.labels), c[0], c[1], rad * 1.25, INK, "Bold", alpha=alpha)

        shown = set()
        if shape:
            age = t - chord.t0
            alpha = min(1.0, age / 0.15)
            grow = _ease_out_back(_clamp01(age / 0.3)) * (1 + 0.12 * pulse)
            for s in range(6):
                f = shape[s]
                if f < 0:
                    c = P(-0.05, string_v(s))
                    text(ctx, "×", c[0], c[1], 16 * u, (1, 1, 1), "Bold", stroke=3 * u, stroke_rgb=INK, alpha=alpha)
                    continue
                dot(s, f, theory.OPEN_STRINGS[s] + f, alpha, grow)
                shown.add((s, f))
        else:
            # No chord: light the single notes where they were most likely played.
            for n in self.ev.notes:
                if n.t0 > t:
                    break
                p = n.pos
                if p is None or t > n.t1 + 0.15 or p in shown:
                    continue
                shown.add(p)
                fade = 1 - (t - n.t1) / 0.15 if t > n.t1 else 1.0
                dot(p[0], p[1], n.midi, fade, _ease_out_back(_clamp01((t - n.t0) / 0.25)))

    # -- piano --
    def _piano(self, ctx, t):
        kb = self.vr.keyboard_at(t)
        u = self.u
        if kb is None:
            self._box_outline(ctx, t)
            self._keyboard_strip(ctx, t)
            return
        d_on = self._since_onset(t)
        pulse = max(0.0, 1 - d_on / 0.35)
        lo, hi = kb.midi_range()
        visible = [m for m in range(lo, hi + 1) if kb.seen(m) and 0 <= kb.x_of(m)[0] < self.W and 0 <= kb.x_of(m)[1] < self.H]
        if self.o.outline and visible:
            wl, wr = kb.key_w(visible[0])[0] - 0.5, kb.key_w(visible[-1])[0] + 0.5
            corners = [kb.point(wl, 0), kb.point(wr, 0), kb.point(wr, 1), kb.point(wl, 1)]
            glow_stroke(ctx, lambda: poly(ctx, corners), ACCENT, 2.0 * u, pulse, self.o.outline, t)
        if not self.o.tutorial:
            return
        speed = self.H * 0.3
        for n in self.ev.notes:
            if n.t0 > t:
                break
            if not kb.has_key(n.midi) or t > n.t1 + 4:
                continue
            rgb = theory.note_rgb(n.midi, 0.62)
            w, black = kb.key_w(n.midi)
            back, front = kb.point(w, 0), kb.point(w, 1)
            up = back - front
            up /= (np.linalg.norm(up) or 1)
            half = np.linalg.norm(kb.point(w + (0.29 if black else 0.36), 0) - back)
            near, far = max(0.0, t - n.t1) * speed, (t - n.t0) * speed
            a, b = back + up * near, back + up * far
            side = np.array([-up[1], up[0]]) * half
            ctx.save()
            for mul, al in ((2.2, 0.15), (1.0, 0.8)):
                ctx.set_source_rgba(*rgb, al)
                poly(ctx, [a - side * mul, b - side * mul, b + side * mul, a + side * mul])
                ctx.fill()
            ctx.restore()
            if t > n.t1:
                continue
            pts = kb.key_polygon(n.midi)
            ctx.save()
            ctx.set_source_rgba(*rgb, 0.65)
            poly(ctx, pts); ctx.fill_preserve()
            ctx.set_source_rgba(1, 1, 1, 0.9)
            ctx.set_line_width(1.5 * u)
            ctx.stroke()
            ctx.restore()
            if self.o.labels and not black:
                lp = kb.point(w, 0.86)
                text(ctx, theory.note_label(n.midi, "letter" if self.o.labels == "octave" else self.o.labels), lp[0], lp[1],
                     max(9 * u, min(18 * u, half * 1.4)), INK, "Bold")

    def _keyboard_strip(self, ctx, t):
        """Fallback when the keys could not be mapped: a glowing keyboard along the bottom."""
        if not self.o.tutorial:
            return
        mids = [n.midi for n in self.ev.notes] or [60]
        lo = max(21, (min(mids) // 12) * 12)
        hi = min(108, (max(mids) // 12 + 1) * 12 - 1)
        while hi - lo < 35:
            lo, hi = max(21, lo - 12), min(108, hi + 12)
        whites = [m for m in range(lo, hi + 1) if m % 12 in theory_whites]
        kh, ww = self.H * 0.13, self.W / len(whites)
        y0 = self.H - kh
        active = {n.midi for n in self.ev.notes if n.t0 <= t <= n.t1}
        ctx.set_source_rgba(0.05, 0.06, 0.09, 0.55)
        ctx.rectangle(0, y0 - 4 * self.u, self.W, kh + 4 * self.u); ctx.fill()
        xs = {}
        for i, m in enumerate(whites):
            xs[m] = i * ww
            ctx.set_source_rgba(*(theory.note_rgb(m, 0.66) if m in active else (0.98, 0.98, 0.99)), 0.95)
            rounded_rect(ctx, i * ww + 1, y0, ww - 2, kh, 4 * self.u); ctx.fill()
        for m in range(lo, hi + 1):
            if m % 12 in theory_whites or m - 1 not in xs:
                continue
            x = xs[m - 1] + ww - ww * 0.3
            ctx.set_source_rgba(*(theory.note_rgb(m, 0.52) if m in active else (0.1, 0.11, 0.15)), 1)
            rounded_rect(ctx, x, y0, ww * 0.6, kh * 0.62, 3 * self.u); ctx.fill()

    # -- drums --
    def _drums(self, ctx, t):
        u = self.u
        pieces = {p.kind: p for p in self.vr.drums}
        shift = self.vr.shift_at(t, self.vr.times[len(self.vr.times) // 2]) if len(self.vr.times) else np.zeros(2)
        box = self.vr.box_at(t)
        for kind in ("kick", "snare", "hat"):
            p = pieces.get(kind)
            hits = [h for h in self.ev.drums if h.kind == kind and h.t <= t]
            age = t - hits[-1].t if hits else 9.0
            flash = max(0.0, 1 - age / 0.3)
            if p is not None:
                cx, cy = p.centre + shift
                rx, ry = p.radii
                if kind != "kick":
                    ry = min(ry, rx * 0.45)
                k = 1 + 0.06 * flash
                if self.o.tutorial and flash:
                    ctx.save(); ctx.translate(cx, cy); ctx.scale(rx * k, ry * k); ctx.arc(0, 0, 1, 0, TAU); ctx.restore()
                    ctx.set_source_rgba(*DRUM_COLORS[kind], 0.35 * flash); ctx.fill()
                if self.o.outline:
                    def ell(cx=cx, cy=cy, rx=rx * k, ry=ry * k):
                        ctx.save(); ctx.translate(cx, cy); ctx.scale(rx, ry); ctx.new_path(); ctx.arc(0, 0, 1, 0, TAU); ctx.restore()
                    glow_stroke(ctx, ell, DRUM_COLORS[kind], 2.0 * u, flash, self.o.outline, t)
        # Hits: impact lines bursting from the rim, and a word.
        i0 = np.searchsorted(self.drum_t, t - 0.8)
        i1 = np.searchsorted(self.drum_t, t + 1e-6)
        for i in range(i0, i1):
            h = self.ev.drums[i]
            a = t - h.t
            p = pieces.get(h.kind)
            rgb = DRUM_COLORS[h.kind]
            if p is not None:
                cx, cy = p.centre + shift
                rx, ry = p.radii
                if h.kind != "kick":
                    ry = min(ry, rx * 0.45)
            elif box is not None:
                rel = {"kick": (0.5, 0.75), "snare": (0.3, 0.5), "hat": (0.75, 0.3)}[h.kind]
                cx, cy = box[0] + rel[0] * (box[2] - box[0]), box[1] + rel[1] * (box[3] - box[1])
                rx = ry = 40 * u
            else:
                cx, cy = self.W * {"kick": 0.5, "snare": 0.3, "hat": 0.72}[h.kind], self.H * {"kick": 0.7, "snare": 0.55, "hat": 0.35}[h.kind]
                rx = ry = 40 * u
            if a < 0.4:
                e = 1 - (1 - a / 0.4) ** 3
                n_lines = {"kick": 14, "snare": 12, "hat": 8}[h.kind]
                ctx.set_line_cap(cairo.LINE_CAP_ROUND)
                ctx.set_source_rgba(*rgb, 1 - e)
                for k in range(n_lines):
                    ang = k / n_lines * TAU + _hash(i, k + 3) * 0.35
                    ln = (46 * (1 - e) + 8) * u
                    ex, ey = rx * (1.08 + 0.5 * e) + ln * 0.2, ry * (1.08 + 0.5 * e) + ln * 0.2
                    ctx.set_line_width((6 * (1 - e) + 1.5) * u)
                    ctx.move_to(cx + math.cos(ang) * ex, cy + math.sin(ang) * ey)
                    ctx.line_to(cx + math.cos(ang) * (ex + ln), cy + math.sin(ang) * (ey + ln))
                    ctx.stroke()
            if a < 0.7:
                pop = _ease_out_back(_clamp01(a / 0.16))
                alpha = 1 - (a - 0.45) / 0.25 if a > 0.45 else 1.0
                words = DRUM_WORDS[h.kind]
                word = words[int(_hash(i, 7) * len(words)) % len(words)]
                size = {"kick": 50, "snare": 40, "hat": 30}[h.kind] * u * pop
                if size > 4:
                    text(ctx, word, cx, cy - ry - 30 * u - a * 40 * u, size, rgb, "Bold", stroke=5 * u, stroke_rgb=INK, alpha=max(0.0, alpha))

    # -- melody instruments and voice --
    def _melody(self, ctx, t):
        box = self._box_outline(ctx, t, (0.45, 0.78, 1.0))
        src = self._source(t)
        if src is None or not self.o.outline:
            return
        d_on = self._since_onset(t)
        for k in range(3):
            a = d_on - k * 0.12
            if 0 <= a <= 0.9:
                e = a / 0.9
                r = (24 + 140 * e) * self.u
                glow_stroke(ctx, lambda r=r: (ctx.new_path(), ctx.arc(src[0], src[1], r, -math.pi * 0.85, -math.pi * 0.15)),
                            (0.45, 0.78, 1.0), 1.4 * self.u * (1 - e) + 0.3, 0, self.o.outline, t)

    def _source(self, t):
        box = self.vr.box_at(t)
        if box is not None:
            return np.array([(box[0] + box[2]) / 2, box[1] + 0.35 * (box[3] - box[1])])
        wr = self.vr.wrists_at(t)
        if len(wr):
            return wr.mean(0)
        return np.array([self.W / 2, self.H * 0.45])

    # -- flying notes --
    def _note_origin(self, n, i):
        nm = self.neck(n.t0)
        if nm is not None:
            p = n.pos
            return nm.fret_point(p[1], p[0]) if p else nm.point(1.45, 0.5)
        if self.mode == "piano":
            kb = self.vr.keyboard_at(n.t0)
            if kb is not None and kb.has_key(n.midi):
                return kb.point(kb.key_w(n.midi)[0], 0)
        src = self._source(n.t0)
        return src + np.array([(_hash(i, 1) - 0.5) * 0.2 * self.W, (_hash(i, 2) - 0.5) * 0.06 * self.H])

    def _flying_notes(self, ctx, t):
        life = 1.8 if self.mode == "guitar" else 2.4
        u = self.u
        i1 = int(np.searchsorted(self.fly_t, t + 1e-6))
        i0 = max(int(np.searchsorted(self.fly_t, t - life)), i1 - 40)
        for i in range(i0, i1):
            n = self.fly[i]
            a = t - n.t0
            x0, y0 = self._note_origin(n, i)
            r1, r2, r3, r4 = (_hash(i, k) for k in (1, 2, 3, 4))
            vx = (r3 - 0.5) * 140 * u
            pos = lambda aa: (x0 + vx * aa + math.sin(aa * 3 + r4 * 6) * 14 * u, y0 - 10 * u - (150 * aa + 25 * aa * aa) * u)
            x, y = pos(a)
            s = (18 + 12 * n.velocity) * u
            pop = _ease_out_back(_clamp01(a / 0.3))
            alpha = (life - a) / 0.6 if a > life - 0.6 else 1.0
            rgb = theory.note_rgb(n.midi, 0.62)
            if self.o.cute:
                ctx.set_source_rgba(1, 0.94, 0.7, 0.55 * alpha)
                for k in range(1, 4):
                    pa = a - k * 0.09
                    if pa < 0:
                        break
                    px, py = pos(pa)
                    star(ctx, px, py + s * 0.4, s * 0.28 * (1 - k * 0.2), pa * 4)
            if pop <= 0.01:
                continue
            ctx.save()
            ctx.translate(x, y)
            ctx.rotate(math.sin(a * 2.2 + r1 * 6) * 0.22)
            ctx.scale(pop, pop)
            ctx.push_group()
            note_glyph(ctx, s, rgb, 2 if r2 < 0.3 else 1, self.o.cute)
            if self.o.labels:
                label = theory.note_label(n.midi, self.o.labels)
                fs = s * 0.95
                w = text_width(label, fs) + fs * 0.7
                h = fs * 1.25
                rounded_rect(ctx, s * 1.25, s * 0.45 - h / 2, w, h, h / 2)
                ctx.set_source_rgb(1, 1, 1); ctx.fill_preserve()
                ctx.set_source_rgb(*rgb); ctx.set_line_width(fs * 0.12); ctx.stroke()
                text(ctx, label, s * 1.25 + w / 2, s * 0.45, fs, INK, "Bold")
            ctx.pop_group_to_source()
            ctx.paint_with_alpha(max(0.0, alpha))
            ctx.restore()

    # -- chord card and timeline --
    def _chord_card(self, ctx, t):
        ci, c = self._chord_at(t)
        if c is None:
            return
        u = self.u
        age = t - c.t0
        d_on = self._since_onset(t)
        bounce = math.sin(d_on / 0.25 * math.pi) * 0.04 if d_on < 0.25 else 0.0
        pop = _ease_out_back(_clamp01(age / 0.35)) * (1 + bounce)
        guitar = self.mode == "guitar"
        shape = theory.guitar_shape(c.label) if guitar else None
        cw, ch = 168 * u, (226 if guitar else 172) * u
        x, y = 24 * u, 24 * u
        rgb = theory.note_rgb(c.label.root, 0.58)
        if self.o.cute and age < 0.9:
            e = 1 - (1 - age / 0.9) ** 3
            for k in range(10):
                ang = k / 10 * TAU + _hash(ci, k) * 0.4
                dd = (60 + 120 * e) * u
                ctx.set_source_rgba(*(rgb if k % 2 else (1, 0.89, 0.48)), 1 - age / 0.9)
                star(ctx, x + cw / 2 + math.cos(ang) * dd, y + ch / 2 + math.sin(ang) * dd, (9 - 4 * e) * u, age * 5 + k)
        ctx.save()
        ctx.translate(x + cw / 2, y + ch / 2)
        ctx.scale(pop, pop)
        ctx.translate(-cw / 2, -ch / 2)
        ctx.set_source_rgba(0, 0, 0, 0.25)
        rounded_rect(ctx, 3 * u, 6 * u, cw, ch, 22 * u); ctx.fill()
        rounded_rect(ctx, 0, 0, cw, ch, 22 * u)
        ctx.set_source_rgba(1, 1, 1, 0.95); ctx.fill_preserve()
        ctx.set_source_rgb(*rgb); ctx.set_line_width(5 * u); ctx.stroke()
        name = c.label.name
        text(ctx, name, cw / 2, 44 * u, (44 if len(name) > 4 else 54) * u, theory.note_rgb(c.label.root, 0.34), "Bold")
        text(ctx, c.label.quality.upper(), cw / 2, 84 * u, 12 * u, (0.37, 0.4, 0.47), "SemiBold")
        if guitar and shape:
            self._mini_fretboard(ctx, cw / 2, 106 * u, 100 * u, 104 * u, shape, rgb)
        elif not guitar:
            self._mini_keys(ctx, 16 * u, 104 * u, cw - 32 * u, 52 * u, c.label)
        ctx.restore()

    def _mini_fretboard(self, ctx, cx, top, w, h, shape, rgb):
        u = self.u
        left, sp, fh = cx - w / 2, w / 5, h / 4
        fretted = [f for f in shape if f > 0]
        base = 1 if not fretted or max(fretted) <= 4 else min(fretted)
        ctx.set_source_rgb(0.54, 0.58, 0.65)
        ctx.set_line_width(1.5 * u)
        for s in range(6):
            ctx.move_to(left + s * sp, top); ctx.line_to(left + s * sp, top + h); ctx.stroke()
        for f in range(5):
            ctx.set_line_width((5 if f == 0 and base == 1 else 1.5) * u)
            ctx.set_source_rgb(*(INK if f == 0 and base == 1 else (0.54, 0.58, 0.65)))
            ctx.move_to(left, top + f * fh); ctx.line_to(left + w, top + f * fh); ctx.stroke()
        if base > 1:
            text(ctx, f"{base}fr", left - 6 * u, top + fh * 0.5, 12 * u, (0.37, 0.4, 0.47), "SemiBold", anchor="rm")
        for s, f in enumerate(shape):
            x = left + s * sp
            if f < 0:
                text(ctx, "×", x, top - 9 * u, 13 * u, (0.54, 0.58, 0.65), "Bold")
            elif f == 0:
                ctx.set_source_rgb(*INK); ctx.set_line_width(1.8 * u)
                ctx.arc(x, top - 9 * u, 4.5 * u, 0, TAU); ctx.stroke()
            else:
                ctx.arc(x, top + (f - base + 0.5) * fh, sp * 0.36, 0, TAU)
                ctx.set_source_rgb(*rgb); ctx.fill_preserve()
                ctx.set_source_rgb(*INK); ctx.set_line_width(1.8 * u); ctx.stroke()

    def _mini_keys(self, ctx, x, y, w, h, label):
        tones = label.tones
        ww = w / 7
        for i, pc in enumerate([0, 2, 4, 5, 7, 9, 11]):
            rounded_rect(ctx, x + i * ww + 1, y, ww - 2, h, 4 * self.u)
            ctx.set_source_rgb(*(theory.note_rgb(pc + 60, 0.64) if pc in tones else (0.95, 0.95, 0.97))); ctx.fill_preserve()
            ctx.set_source_rgb(0.79, 0.81, 0.85); ctx.set_line_width(1); ctx.stroke()
        for pc, after in ((1, 1), (3, 2), (6, 4), (8, 5), (10, 6)):
            rounded_rect(ctx, x + after * ww - ww * 0.3, y, ww * 0.6, h * 0.6, 3 * self.u)
            ctx.set_source_rgb(*(theory.note_rgb(pc + 60, 0.48) if pc in tones else INK)); ctx.fill()

    def _timeline(self, ctx, t):
        """Current and upcoming chords along the bottom, so players can follow along."""
        chords = self.ev.chords
        if not chords:
            return
        u = self.u
        i = int(np.searchsorted(self.chord_t, t + 1e-6)) - 1
        upcoming = [k for k in range(max(0, i), min(len(chords), max(0, i) + 4))]
        if not upcoming:
            return
        h = 34 * u
        y = self.H - h - 18 * u
        if self.mode == "piano" and self.vr.keyboard is None:
            y -= self.H * 0.13  # stay above the fallback keyboard strip
        sizes = [max(54 * u, text_width(chords[k].label.name, 20 * u) + 30 * u) for k in upcoming]
        total = sum(sizes) + 10 * u * (len(sizes) - 1)
        x = (self.W - total) / 2
        for k, w in zip(upcoming, sizes):
            c = chords[k]
            current = c.t0 <= t <= c.t1
            rgb = theory.note_rgb(c.label.root, 0.6)
            alpha = 0.95 if current else 0.6
            rounded_rect(ctx, x, y, w, h, h / 2)
            ctx.set_source_rgba(*(rgb if current else (0.08, 0.09, 0.13)), alpha); ctx.fill_preserve()
            ctx.set_source_rgba(*rgb, 1); ctx.set_line_width(2 * u); ctx.stroke()
            text(ctx, c.label.name, x + w / 2, y + h / 2, 20 * u, INK if current else (1, 1, 1), "Bold")
            if current and c.t1 > c.t0:
                prog = _clamp01((t - c.t0) / (c.t1 - c.t0))
                ctx.set_source_rgba(1, 1, 1, 0.85)
                rounded_rect(ctx, x + h / 2, y + h - 5 * u, (w - h) * prog, 3 * u, 1.5 * u); ctx.fill()
            x += w + 10 * u


theory_whites = {0, 2, 4, 5, 7, 9, 11}
