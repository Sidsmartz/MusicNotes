"""Web app: upload a video, follow progress, watch and download the result."""
from __future__ import annotations

import os
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import media
from ..pipeline import Options, process
from ..render import RenderOptions

STATIC = Path(__file__).resolve().parent / "static"
DATA = Path(os.environ.get("MUSICNOTES_DATA", "data")).resolve() / "jobs"
MAX_UPLOAD_MB = int(os.environ.get("MUSICNOTES_MAX_UPLOAD_MB", "500"))
MAX_SECONDS = float(os.environ.get("MUSICNOTES_MAX_SECONDS", "300"))
KEEP_HOURS = float(os.environ.get("MUSICNOTES_KEEP_HOURS", "24"))
VIDEO_TYPES = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".3gp"}


@dataclass
class Job:
    id: str
    name: str
    status: str = "queued"  # queued | running | done | failed
    stage: str = "queued"
    progress: float = 0.0
    message: str = "Waiting for a free worker"
    error: str = ""
    summary: dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)

    def public(self) -> dict:
        return {k: getattr(self, k) for k in ("id", "name", "status", "stage", "progress", "message", "error", "summary")}


class Worker:
    """Runs one job at a time; the models are loaded once and shared."""

    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.pool = ThreadPoolExecutor(max_workers=1)
        self._models = None
        self._lock = threading.Lock()

    def models(self):
        with self._lock:
            if self._models is None:
                from ..vision.detector import InstrumentDetector, PoseDetector

                self._models = (InstrumentDetector(), PoseDetector())
            return self._models

    def submit(self, job: Job, src: Path, options: Options):
        self.jobs[job.id] = job
        self.pool.submit(self._run, job, src, options)

    def _run(self, job: Job, src: Path, options: Options):
        from ..pipeline import _overall

        job.status, job.message = "running", "Starting"
        try:
            detector, pose = self.models()

            def progress(stage, p, msg):
                job.stage, job.progress, job.message = stage, round(_overall(stage, p), 3), msg

            job.summary = process(str(src), str(src.parent / "output.mp4"), options, progress, detector, pose)
            job.status, job.stage, job.progress, job.message = "done", "done", 1.0, "Ready"
        except media.MediaError as e:
            job.status, job.error = "failed", str(e)
        except Exception:  # noqa: BLE001 - surface any failure to the user
            traceback.print_exc()
            job.status, job.error = "failed", "Something went wrong while processing this video. Try a shorter clip or another format."
        finally:
            src.unlink(missing_ok=True)


def _cleanup():
    if not DATA.exists():
        return
    cutoff = time.time() - KEEP_HOURS * 3600
    for d in DATA.iterdir():
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)


def create_app() -> FastAPI:
    DATA.mkdir(parents=True, exist_ok=True)
    _cleanup()
    app = FastAPI(title="MusicNotes")
    worker = Worker()
    app.state.worker = worker

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.post("/api/jobs")
    async def create_job(
        file: UploadFile = File(...),
        instrument: str = Form("auto"),
        notes: bool = Form(True),
        tutorial: bool = Form(True),
        outline: str = Form("neon"),
        chord_card: bool = Form(True),
        timeline: bool = Form(True),
        labels: str = Form("letter"),
        cute: bool = Form(True),
        shake: bool = Form(True),
        sensitivity: float = Form(0.5),
        quality: int = Form(1280),
    ):
        ext = Path(file.filename or "").suffix.lower()
        if ext not in VIDEO_TYPES:
            raise HTTPException(400, "Upload a video file: MP4, MOV, WebM, MKV or AVI.")
        if instrument not in ("auto", "guitar", "piano", "drums", "melody"):
            raise HTTPException(400, "Unknown instrument.")
        job = Job(uuid.uuid4().hex[:12], Path(file.filename).stem[:80] or "video")
        folder = DATA / job.id
        folder.mkdir(parents=True)
        src = folder / f"input{ext}"
        size = 0
        with src.open("wb") as out:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_MB << 20:
                    shutil.rmtree(folder, ignore_errors=True)
                    raise HTTPException(413, f"That file is over {MAX_UPLOAD_MB} MB. Trim the video and try again.")
                out.write(chunk)
        try:
            info = media.probe(str(src))
        except media.MediaError as e:
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(400, str(e))
        if info.duration > MAX_SECONDS:
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(400, f"Videos can be up to {int(MAX_SECONDS // 60)} minutes long; this one is {info.duration / 60:.1f}.")
        if not info.has_audio or not info.has_video:
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(400, "The video needs both picture and sound.")
        opts = Options(
            instrument=instrument, sensitivity=max(0.0, min(1.0, sensitivity)), max_side=max(480, min(1920, quality)),
            render=RenderOptions(notes=notes, tutorial=tutorial, outline=outline if outline in ("neon", "dashed") else "",
                                 chord_card=chord_card, timeline=timeline,
                                 labels=labels if labels in ("letter", "octave", "solfege") else "", cute=cute, shake=shake),
        )
        worker.submit(job, src, opts)
        return job.public()

    def get(job_id: str) -> Job:
        job = worker.jobs.get(job_id)
        if not job:
            raise HTTPException(404, "That job does not exist (finished jobs are kept for a day).")
        return job

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        return JSONResponse(get(job_id).public())

    @app.get("/api/jobs/{job_id}/video")
    def job_video(job_id: str, download: bool = False):
        job = get(job_id)
        path = DATA / job.id / "output.mp4"
        if job.status != "done" or not path.exists():
            raise HTTPException(409, "The video is not ready yet.")
        name = f"{job.name}-musicnotes.mp4"
        return FileResponse(path, media_type="video/mp4", filename=name if download else None,
                            content_disposition_type="attachment" if download else "inline")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app

