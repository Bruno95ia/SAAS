# src/saas/clipper.py
from __future__ import annotations
import subprocess, shlex
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

CLIPS_DIR = Path("runs/clips")
CLIPS_DIR.mkdir(parents=True, exist_ok=True)

def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in s)

def make_clip_filename(camera_id: str, when: Optional[datetime] = None, ext: str = "mp4") -> Path:
    when = when or datetime.utcnow()
    name = f"{_slug(camera_id)}_{when.strftime('%Y%m%dT%H%M%S')}.{ext}"
    return CLIPS_DIR / name

def has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        return False

def cut_with_ffmpeg(src_video: str, start_sec: float, duration_sec: float, out_path: Path) -> None:
    """
    Corta sem re-encode quando possível (stream copy). Para MP4/H264 é perfeito.
    """
    cmd = f'ffmpeg -y -ss {start_sec:.3f} -i {shlex.quote(src_video)} -t {duration_sec:.3f} -c copy {shlex.quote(str(out_path))}'
    # alguns containers não suportam -c copy, então tentamos re-encode como fallback
    p = subprocess.run(cmd, shell=True)
    if p.returncode != 0:
        cmd = f'ffmpeg -y -ss {start_sec:.3f} -i {shlex.quote(src_video)} -t {duration_sec:.3f} -c:v libx264 -preset veryfast -crf 23 -c:a aac {shlex.quote(str(out_path))}'
        subprocess.check_call(cmd, shell=True)

def cut_with_moviepy(src_video: str, start_sec: float, duration_sec: float, out_path: Path) -> None:
    from moviepy.editor import VideoFileClip  # lazy import
    end = start_sec + duration_sec
    with VideoFileClip(src_video).subclip(start_sec, end) as clip:
        clip.write_videofile(str(out_path), codec="libx264", audio_codec="aac", verbose=False, logger=None)

def save_clip_from_file(
    src_video: str,
    event_ts_sec: float,
    pre_sec: float = 5.0,
    post_sec: float = 5.0,
    camera_id: str = "cam",
    api_base_url: str = "http://127.0.0.1:8000",
) -> Tuple[str, str]:
    """
    Corta [pre_sec .. post_sec] ao redor de event_ts_sec a partir de um arquivo de vídeo local.
    Retorna (file_path_str, clip_url).
    """
    start = max(0.0, event_ts_sec - pre_sec)
    duration = pre_sec + post_sec
    out_path = make_clip_filename(camera_id)

    if has_ffmpeg():
        cut_with_ffmpeg(src_video, start, duration, out_path)
    else:
        cut_with_moviepy(src_video, start, duration, out_path)

    clip_url = f"{api_base_url.rstrip('/')}/clips/{out_path.name}"
    return str(out_path), clip_url