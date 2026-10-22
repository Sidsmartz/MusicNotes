"""Command line: process a video, or start the web app.

    python -m musicnotes input.mp4 -o output.mp4 [--instrument guitar]
    python -m musicnotes serve [--port 8000]
"""
from __future__ import annotations

import argparse
import sys
import time

from . import pipeline
from .render import RenderOptions


def _process(argv):
    p = argparse.ArgumentParser(prog="musicnotes", description="Draw notes, chords and hits onto a video of someone playing.")
    p.add_argument("input")
    p.add_argument("-o", "--output", help="output MP4 (default: <input>-musicnotes.mp4)")
    p.add_argument("--instrument", default="auto", choices=["auto", "guitar", "piano", "drums", "melody"])
    p.add_argument("--sensitivity", type=float, default=0.5, help="0..1, higher catches quieter notes")
    p.add_argument("--size", type=int, default=1280, help="longest side of the output in pixels")
    p.add_argument("--outline", default="neon", choices=["neon", "dashed", "off"])
    p.add_argument("--labels", default="letter", choices=["letter", "octave", "solfege", "off"])
    for flag in ("notes", "tutorial", "chord-card", "timeline", "cute", "shake"):
        p.add_argument(f"--no-{flag}", action="store_true", help=f"turn off {flag.replace('-', ' ')}")
    p.add_argument("--json", action="store_true", help="print the analysis summary as JSON")
    a = p.parse_args(argv)
    out = a.output or a.input.rsplit(".", 1)[0] + "-musicnotes.mp4"
    opts = pipeline.Options(
        instrument=a.instrument, sensitivity=a.sensitivity, max_side=a.size,
        render=RenderOptions(notes=not a.no_notes, tutorial=not a.no_tutorial, outline="" if a.outline == "off" else a.outline,
                             chord_card=not a.no_chord_card, timeline=not a.no_timeline,
                             labels="" if a.labels == "off" else a.labels, cute=not a.no_cute, shake=not a.no_shake),
    )
    last = [0.0]

    def progress(stage, frac, msg):
        if time.time() - last[0] > 0.5 or frac >= 1:
            sys.stderr.write(f"\r{msg:<40} {round(pipeline._overall(stage, frac) * 100):3d}%")
            sys.stderr.flush()
            last[0] = time.time()

    summary = pipeline.process(a.input, out, opts, progress)
    sys.stderr.write("\n")
    if a.json:
        print(pipeline.main_json(summary))
    else:
        print(f"{out}\n{summary['instrument_name']} ({summary['why']}); "
              f"{len(summary['chords'])} chords, {summary['notes']} notes, made in {summary['seconds']} s")


def _serve(argv):
    p = argparse.ArgumentParser(prog="musicnotes serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args(argv)
    import uvicorn

    uvicorn.run("musicnotes.web.app:create_app", factory=True, host=a.host, port=a.port)


def main():
    argv = sys.argv[1:]
    if argv[:1] == ["serve"]:
        _serve(argv[1:])
    else:
        _process(argv)


if __name__ == "__main__":
    main()
