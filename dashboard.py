"""Serve the original Aether dashboard and connect it to video inference."""
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import math
import cv2
from pathlib import Path

from fastapi import File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def mount_dashboard(app, project_dir, process_video):
    root = Path(project_dir)
    # Keep outputs for video range requests; clean up on process exit.
    storage = tempfile.TemporaryDirectory(prefix="aether-dashboard-")
    app.state.dashboard_storage = storage
    output_dir = Path(storage.name)
    inference_lock = threading.Lock()
    app.mount("/results", StaticFiles(directory=output_dir), name="results")
    app.mount("/thumbnails", StaticFiles(directory=root / "thumbnails"), name="thumbnails")

    @app.get("/")
    def dashboard():
        return FileResponse(root / "aether_dashboard.html")

    @app.get("/dashboard.js")
    def dashboard_script():
        return FileResponse(root / "dashboard.js", media_type="text/javascript")

    @app.post("/analyze")
    def analyze(video: UploadFile = File(...)):
        started = time.monotonic()
        output = None
        try:
            with tempfile.TemporaryDirectory(prefix="aether-upload-") as folder:
                source = Path(folder) / "input.mp4"
                with source.open("wb") as target:
                    shutil.copyfileobj(video.file, target)
                # The TFLite interpreter is shared by requests.
                with inference_lock:
                    output, summary = process_video(str(source))
                if not output:
                    raise HTTPException(422, summary)
                capture = cv2.VideoCapture(output)
                fps = capture.get(cv2.CAP_PROP_FPS)
                capture.release()
                fps = fps if math.isfinite(fps) and fps > 0 else 25.0
                name = f"{uuid.uuid4().hex}.mp4"
                # OpenCV's mp4v output does not play in all browsers.
                subprocess.run([
                    "ffmpeg", "-y", "-loglevel", "error", "-i", output,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(output_dir / name),
                ], check=True, capture_output=True, timeout=180)
            events = []
            for line in summary.splitlines()[1:]:
                confidence = re.search(r"Conf: ([0-9.]+)", line)
                frame = re.search(r"^Frame (\d+)", line)
                events.append({"description": line,
                               "seconds": int(frame[1]) / fps if frame else None,
                               "confidence": float(confidence[1]) if confidence else None,
                               "fall": "FALL" in line and "NO_FALL" not in line})
            return {"video_url": f"/results/{name}", "summary": summary,
                    "events": events, "elapsed": round(time.monotonic() - started, 1)}
        finally:
            video.file.close()
            if output:
                Path(output).unlink(missing_ok=True)
