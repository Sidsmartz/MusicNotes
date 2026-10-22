"""End to end: video in, video with effects out."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from . import audio, media, theory
from .render import Overlay, RenderOptions
from .vision.video import KIND_NAMES, VisionResult, align_keyboard_octave, analyze_video

Progress = Callable[[str, float, str], None]  # stage, fraction 0..1, message

STAGES = [("listen", 0.25), ("look", 0.3), ("render", 0.45)]


@dataclass
class Options:
    instrument: str = "auto"  # auto | guitar | piano | drums | melody
    sensitivity: float = 0.5
    max_side: int = 1280
    render: RenderOptions = field(default_factory=RenderOptions)


def _overall(stage: str, p: float) -> float:
    done = 0.0
    for name, weight in STAGES:
        if name == stage:
            return done + weight * max(0.0, min(1.0, p))
        done += weight
    return 1.0


def resolve_mode(choice: str, vision: VisionResult, guess: audio.AudioGuess) -> tuple[str, str]:
    """Which instrument to animate, and why."""
    if choice in KIND_NAMES:
        return choice, "chosen by you"
    if vision.kind and vision.coverage >= 0.25:
        return vision.kind, f"seen in {round(vision.coverage * 100)}% of the video"
    return guess.mode, f"from the sound alone: {guess.reason}"


def process(src: str, dst: str, options: Options | None = None, progress: Progress | None = None,
            detector=None, pose=None) -> dict:
    options = options or Options()
    report = progress or (lambda stage, p, msg: None)
    started = time.time()

    info = media.probe(src)
    if not info.has_audio:
        raise media.MediaError("This video has no sound, so there is nothing to listen to.")
    if not info.has_video:
        raise media.MediaError("This file has no picture. Upload a video of you playing.")

    report("listen", 0.0, "Listening to the music")
    y = media.read_audio(src, audio.SR)
    A = audio.analyze(y, progress=lambda p, msg="": report("listen", p, msg or "Listening to the music"))

    report("look", 0.0, "Looking for the instrument")
    forced = options.instrument if options.instrument in KIND_NAMES else None
    vision = analyze_video(src, info, forced_kind=forced, max_side=options.max_side, detector=detector, pose=pose,
                           progress=lambda p, msg="": report("look", p, msg or "Looking for the instrument"))
    mode, why = resolve_mode(options.instrument, vision, A.guess)
    if vision.kind != mode:
        # The camera saw something else (or nothing): keep the box track only if it is this instrument.
        vision = VisionResult(vision.size, None, "", 0.0, 0.0, vision.times, vision.boxes[:0], wrists=vision.wrists) \
            if vision.kind else vision
    events = audio.build_events(A, mode, options.sensitivity)
    if mode == "piano":
        align_keyboard_octave(vision, events.notes)

    size = vision.size
    overlay = Overlay(size, events, vision, options.render)
    total = max(1, int(info.duration * info.fps))
    writer = media.FrameWriter(dst, size, info.fps, src)
    report("render", 0.0, "Drawing the effects")
    try:
        for i, frame in enumerate(media.FrameReader(src, size, info.fps)):
            writer.write(overlay.draw(frame, i / info.fps))
            if i % 10 == 0:
                report("render", i / total, "Drawing the effects")
    finally:
        writer.close()
    report("render", 1.0, "Done")

    summary = {
        "instrument": mode,
        "instrument_name": KIND_NAMES[mode],
        "why": why,
        "seen": {"kind": vision.kind, "label": vision.label, "confidence": round(vision.confidence, 3),
                 "coverage": round(vision.coverage, 3)},
        "geometry": {
            "neck": vision.neck is not None, "frets_fitted": round(vision.neck_fitted, 2),
            "keyboard": vision.keyboard is not None, "drum_pieces": [p.kind for p in vision.drums],
        },
        "audio_guess": {"mode": A.guess.mode, "reason": A.guess.reason},
        "tempo": round(A.tempo),
        "duration": round(info.duration, 2),
        "chords": [{"name": c.label.name, "t0": round(c.t0, 2), "t1": round(c.t1, 2)} for c in events.chords],
        "notes": len(events.notes),
        "drums": {k: sum(1 for d in events.drums if d.kind == k) for k in ("kick", "snare", "hat")},
        "key_notes": sorted({theory.NAMES[n.midi % 12] for n in events.notes}),
        "options": {**asdict(options.render), "instrument": options.instrument, "sensitivity": options.sensitivity},
        "seconds": round(time.time() - started, 1),
    }
    return summary


def main_json(summary: dict) -> str:
    return json.dumps(summary, indent=2)
