import json
import subprocess
import time

import cv2
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from musicnotes import media
from musicnotes.pipeline import Options, process
from tests import drawings, synth


def _make_video(tmp_path, image, audio, seconds=4.0, name="clip.mp4"):
    img_path, wav_path, out = tmp_path / "frame.png", tmp_path / "audio.wav", tmp_path / name
    cv2.imwrite(str(img_path), image)
    sf.write(wav_path, audio[: int(seconds * synth.SR)], synth.SR)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", str(img_path), "-i", str(wav_path), "-t", str(seconds),
                    "-r", "24", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)], check=True)
    return out


@pytest.fixture(scope="module")
def guitar_clip(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("guitar")
    img, *_ = drawings.guitar(10.0, size=(960, 540))
    return _make_video(tmp, img, synth.guitar(repeats=1)[0])


def test_guitar_video_end_to_end(guitar_clip, tmp_path):
    out = tmp_path / "out.mp4"
    summary = process(str(guitar_clip), str(out), Options(instrument="guitar", max_side=640))
    assert summary["instrument"] == "guitar"
    assert summary["geometry"]["neck"], summary["geometry"]
    assert [c["name"] for c in summary["chords"]][:2] == ["A", "D"]
    info = media.probe(str(out))
    assert info.has_audio and info.has_video
    assert abs(info.duration - 4.0) < 0.3
    assert max(info.width, info.height) == 640
    # The overlay actually changed the picture.
    src = next(iter(media.FrameReader(str(guitar_clip), (info.width, info.height), 24)))
    dst = next(iter(media.FrameReader(str(out), (info.width, info.height), 24)))
    assert np.abs(src.astype(int) - dst.astype(int)).mean() > 1.0


def test_keyboard_video_lights_keys(tmp_path):
    img, *_ = drawings.keyboard(size=(960, 540))
    clip = _make_video(tmp_path, img, synth.piano(repeats=1)[0])
    summary = process(str(clip), str(tmp_path / "out.mp4"), Options(instrument="piano", max_side=640))
    assert summary["geometry"]["keyboard"], summary
    assert summary["chords"][0]["name"] == "C"


def test_web_upload_and_download(guitar_clip, tmp_path, monkeypatch):
    monkeypatch.setenv("MUSICNOTES_DATA", str(tmp_path))
    import importlib

    from musicnotes.web import app as webapp

    importlib.reload(webapp)
    client = TestClient(webapp.create_app())
    with open(guitar_clip, "rb") as f:
        r = client.post("/api/jobs", files={"file": ("clip.mp4", f, "video/mp4")},
                        data={"instrument": "guitar", "quality": "640", "outline": "dashed"})
    assert r.status_code == 200, r.text
    job = r.json()
    for _ in range(240):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert job["status"] == "done", job
    assert job["summary"]["options"]["outline"] == "dashed"
    video = client.get(f"/api/jobs/{job['id']}/video?download=true")
    assert video.status_code == 200
    assert video.headers["content-type"] == "video/mp4"
    assert "attachment" in video.headers["content-disposition"]
    assert len(video.content) > 10_000


def test_web_rejects_non_video(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSICNOTES_DATA", str(tmp_path))
    from musicnotes.web import app as webapp

    client = TestClient(webapp.create_app())
    r = client.post("/api/jobs", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400
    assert "video" in json.loads(r.text)["detail"].lower()
