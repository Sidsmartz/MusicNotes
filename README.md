# MusicNotes

Upload a video of yourself playing. MusicNotes finds your instrument in the picture, works out what you played from the sound, and draws the music onto the instrument itself. You get an MP4 with the effects drawn into the frames and your original audio.

- **Guitar:** the neck gets a neon outline, with fret numbers along it. The current chord's shape lights up on your real frets, each dot labelled with its note name. The strings shimmer on every strum, and note characters fly out of the fret they were played on. A chord card shows the name and diagram, and a timeline along the bottom shows the current and upcoming chords.
- **Piano and keyboards:** the keys you play light up on the real keyboard, labelled, with light bars rising out of them. You also get the chord card and timeline.
- **Drums:** the kit pieces are outlined. A hit flashes the drum that was struck and bursts impact lines from its rim, with a BOOM, CRACK or tss. Kicks shake the frame.
- **Melody instruments** (voice, violin, sax, trumpet, flute…): the instrument is outlined, and notes and sound waves come out of it.

No generative AI is involved, and nothing is invented. The app detects what is there and draws hand-designed effects on it.

## Run it

You need Python 3.10+ and ffmpeg.

```sh
pip install -r requirements.txt
python -m musicnotes serve            # web app on http://127.0.0.1:8000
python -m musicnotes clip.mp4 -o out.mp4 --instrument guitar   # or from the command line
```

With Docker:

```sh
docker build -t musicnotes .
docker run -p 8000:8000 musicnotes
```

The web app takes uploads up to 500 MB and 5 minutes long. Finished videos are kept for 24 hours. Change these with `MUSICNOTES_MAX_UPLOAD_MB`, `MUSICNOTES_MAX_SECONDS` and `MUSICNOTES_KEEP_HOURS`. Everything runs on the CPU, and processing takes roughly two to four times the clip's length.

`python -m musicnotes --help` lists every effect switch (`--no-notes`, `--outline dashed`, `--labels solfege`, …).

## How it works

```
video ──► audio ──► onsets, beats, notes, chords, drum hits ──┐
   │                                                          ├──► scene ──► frames with effects ──► MP4 (+ original audio)
   └────► frames (3 per second) ──► instrument + hands ──► neck / keys / kit, smoothed over time ──┘
```

**Listening** (`musicnotes/audio.py`, librosa and numpy):
- **Onsets and tempo:** spectral flux, then beat tracking.
- **Notes:** up to six at once. Harmonic summation on a whitened spectrum finds each note, then every harmonic of that note up to 6 kHz is cancelled so it cannot come back as a fake note. Notes are tracked over time with hysteresis.
- **Chords:** a CQT chroma of the harmonic part of the sound is matched against major, minor, 7, m7 and maj7 templates. A Viterbi pass smooths the result, and a no-chord state handles silence.
- **Drums:** band-wise flux of the percussive part sorts each hit into kick, snare or hi-hat.

**Looking** (`musicnotes/vision/`):
- **Instrument detection:** a YOLOv8 detector trained on Open Images V7. It knows guitars, banjos, pianos, keyboards, organs, drums, violins, cellos, saxophones, trumpets, flutes and more. The instrument seen most consistently over the video is chosen; your choice in the app overrides it.
- **Hands:** YOLOv8-pose finds the player's wrists.
- **Guitar neck:** the dominant direction of long straight lines (strings and neck edges) gives the neck's axis. The narrow end of the instrument is the neck. The app then fits the fret law, fret *n* at `scale × (1 − 2^(−n/12))` from the nut, to the edges across the neck. The fit rewards a strong edge at the nut and penalises fret-like edges beyond it. When the frets are not clear, typical guitar proportions are used instead.
- **Keyboard:** a morphological closing isolates the black keys. A line fit with outlier rejection finds the row they sit in. The short/long gap pattern (C# D# · F# G# A# ·) names each black key, and a 1-D perspective fit places every key, black and white. The octave is chosen so the notes being played land under the player's hands.
- **Drum kit:** drum boxes collected over the whole video are clustered, and the pieces are named kick, snare or hi-hat by their layout.
- **Smoothing:** all geometry is smoothed over time with outlier rejection, so one bad frame does not make the overlay jump.

**Drawing** (`musicnotes/render.py`): anti-aliased vector graphics with Cairo, and text in the Fredoka font. Every frame is a pure function of the playback time, so the effects stay in sync with the audio.

## How well it works

`scripts/benchmark.py` downloads real photos from the Open Images V7 validation set and checks the vision side against them. It uses up to 40 photos per class, keeping only real instruments that are not cut off and fill at least 8% of the frame:

| | Photos | Instrument found (IoU ≥ 0.5) | Geometry |
| --- | --- | --- | --- |
| Guitars, ukuleles, banjos, basses | 40 | 37 | neck found on 36; frets fitted from the image on 20, typical proportions on the rest. By eye, about 5 in 6 overlays sit on the actual neck |
| Pianos, keyboards, organs | 40 | 36 | keys mapped on 14 (many photos show closed lids, distant grands, organs or harpsichords) |
| Drum kits | 27 | 19 | n/a |

These are single photos. A video is analysed over many frames and smoothed, which is more forgiving.

The tests (`pytest`) cover audio analysis on synthesised guitar, piano and drum clips with known content, and neck and keyboard geometry on drawings with exactly known frets and keys. They also run the whole pipeline end to end and exercise the web API.

## Limits

- **One instrument in the frame works best.** With a full band, the app follows the most prominent instrument and the sound analysis hears everything at once.
- **Guitar chord shapes** are the standard shape for the detected chord, not necessarily the voicing you played. Single notes are placed at the most likely fret near where your hand was.
- **Fret alignment** needs the neck reasonably visible and not heavily foreshortened. With fast camera moves or a hand covering most of the neck, positions fall back to proportions and can be off by a fret or two.
- **Pianos:** the keys have to be visible from roughly above or in front. Very steep angles, closed lids and reversed-colour keys (harpsichords) are not mapped; the app shows a keyboard strip at the bottom of the frame instead.
- **Drums:** the detector only knows "Drum", so pieces are named by layout. Cymbals are not detected, and the hi-hat effect is placed by layout.
- **Phone videos** with rotation metadata are handled. Variable frame rate videos are converted to constant frame rate.

## Project layout

```
musicnotes/
  audio.py            listening: onsets, notes, chords, drums, instrument guess
  vision/detector.py  YOLOv8 instrument detector and pose model (onnxruntime)
  vision/guitar.py    guitar neck and fret model
  vision/piano.py     keyboard and key model
  vision/drums.py     drum kit layout
  vision/video.py     sampling, tracking and smoothing over the video
  render.py           the effects
  pipeline.py         video in, video out
  media.py            ffmpeg reading and writing
  web/                FastAPI app and the upload page
models/               ONNX models (see models/README.md for the AGPL-3.0 licence)
scripts/              benchmark and model export
tests/
```

## Licences

- The code is in this repository.
- The detector and pose models are Ultralytics YOLOv8 (AGPL-3.0); see `models/README.md`.
- The Fredoka font is under the SIL Open Font License (`musicnotes/assets/fonts/OFL.txt`).
