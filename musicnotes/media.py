"""Reading and writing video through ffmpeg."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np


class MediaError(RuntimeError):
    pass


@dataclass
class VideoInfo:
    width: int  # display size, after rotation metadata is applied
    height: int
    fps: float
    duration: float
    has_audio: bool
    has_video: bool


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise MediaError(f"{tool} is not installed. Install ffmpeg and try again.")
    return path


def probe(path: str) -> VideoInfo:
    cmd = [_require("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise MediaError("This file could not be read as a video.")
    data = json.loads(res.stdout)
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video" and s.get("disposition", {}).get("attached_pic", 0) == 0), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration = float(data.get("format", {}).get("duration") or (v or a or {}).get("duration") or 0)
    if v is None:
        return VideoInfo(0, 0, 30.0, duration, a is not None, False)
    w, h = int(v["width"]), int(v["height"])
    rotation = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(float(sd["rotation"]))
    rotation = int(v.get("tags", {}).get("rotate", rotation) or rotation)
    if abs(rotation) % 180 == 90:
        w, h = h, w
    num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    if not 1 <= fps <= 240:
        fps = 30.0
    return VideoInfo(w, h, fps, duration, a is not None, True)


def output_size(info: VideoInfo, max_side: int) -> tuple[int, int]:
    s = min(1.0, max_side / max(info.width, info.height))
    # Even dimensions for yuv420p.
    return int(round(info.width * s / 2)) * 2, int(round(info.height * s / 2)) * 2


def read_audio(path: str, sr: int) -> np.ndarray:
    cmd = [_require("ffmpeg"), "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise MediaError("The sound in this video could not be read.")
    return np.frombuffer(res.stdout, dtype=np.float32).copy()


class FrameReader:
    """Decodes frames as BGR arrays at a fixed size and frame rate."""

    def __init__(self, path: str, size: tuple[int, int], fps: float):
        self.w, self.h = size
        cmd = [_require("ffmpeg"), "-v", "error", "-i", path, "-an", "-vf", f"fps={fps},scale={self.w}:{self.h}:flags=area",
               "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=self.w * self.h * 3 * 4)
        self.frame_bytes = self.w * self.h * 3

    def __iter__(self):
        while True:
            buf = self.proc.stdout.read(self.frame_bytes)
            if len(buf) < self.frame_bytes:
                break
            yield np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
        self.close()

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()


class FrameWriter:
    """Encodes BGR frames to H.264 and copies the source's audio in."""

    def __init__(self, path: str, size: tuple[int, int], fps: float, audio_from: str | None, crf: int = 20):
        w, h = size
        cmd = [_require("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-"]
        if audio_from:
            cmd += ["-i", audio_from, "-map", "0:v", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray):
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace")
        if self.proc.wait() != 0:
            raise MediaError(f"Encoding the video failed: {err.strip()[-300:]}")
