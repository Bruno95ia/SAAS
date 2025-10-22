"""Ferramentas para geração de clipes a partir de vídeos ou buffers RTSP."""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2


CLIPS_DIR = Path("runs/clips")
CLIPS_DIR.mkdir(parents=True, exist_ok=True)


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in text)


def make_clip_filename(camera_id: str, when: Optional[datetime] = None, ext: str = "mp4") -> Path:
    when = when or datetime.utcnow()
    name = f"{_slug(camera_id)}_{when.strftime('%Y%m%dT%H%M%S')}"
    return CLIPS_DIR / f"{name}.{ext}"


def has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except FileNotFoundError:
        return False


def _run_ffmpeg(cmd: Sequence[str]) -> None:
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg não encontrado. Instale o utilitário para operações com clipes.") from exc


def cut_with_ffmpeg(src_video: str, start_sec: float, duration_sec: float, out_path: Path) -> None:
    """Corta um trecho usando FFmpeg (preferencial quando disponível)."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        src_video,
        "-t",
        f"{duration_sec:.3f}",
        "-c",
        "copy",
        str(out_path),
    ]
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        # Alguns containers (ex.: m4s) não permitem stream copy. Re-encode como fallback.
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            src_video,
            "-t",
            f"{duration_sec:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        _run_ffmpeg(cmd)


def cut_with_cv2(src_video: str, start_sec: float, duration_sec: float, out_path: Path) -> None:
    """Fallback de corte usando OpenCV quando FFmpeg não está disponível."""
    cap = cv2.VideoCapture(src_video)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {src_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Não foi possível criar o arquivo de saída: {out_path}")

    start_frame = max(0, int(start_sec * fps))
    total_frames = int(duration_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        out.write(frame)

    cap.release()
    out.release()


def cut_with_moviepy(src_video: str, start_sec: float, duration_sec: float, out_path: Path) -> None:
    from moviepy.editor import VideoFileClip  # Lazy import para evitar custo se não estiver instalado

    end = start_sec + duration_sec
    with VideoFileClip(src_video).subclip(start_sec, end) as clip:
        clip.write_videofile(str(out_path), codec="libx264", audio_codec="aac", verbose=False, logger=None)


def save_clip_from_file(
    src_video: str,
    event_ts_sec: float,
    pre_sec: float = 5.0,
    post_sec: float = 5.0,
    camera_id: str = "cam",
    api_base_url: Optional[str] = "http://127.0.0.1:8000",
) -> Tuple[str, str]:
    """Corta `[event_ts - pre, event_ts + post]` de um arquivo de vídeo local."""

    start = max(0.0, event_ts_sec - pre_sec)
    duration = pre_sec + post_sec
    out_path = make_clip_filename(camera_id)

    if has_ffmpeg():
        try:
            cut_with_ffmpeg(src_video, start, duration, out_path)
        except RuntimeError:
            cut_with_cv2(src_video, start, duration, out_path)
    else:
        try:
            cut_with_cv2(src_video, start, duration, out_path)
        except RuntimeError:
            try:
                cut_with_moviepy(src_video, start, duration, out_path)
            except ImportError as exc:  # pragma: no cover - dependência opcional
                raise RuntimeError(
                    "Não há suporte para corte de clipes (sem FFmpeg nem MoviePy instalados)."
                ) from exc

    if api_base_url:
        base = api_base_url.rstrip("/")
        clip_url = f"{base}/clips/{out_path.name}" if base else f"/clips/{out_path.name}"
    else:
        clip_url = f"clips/{out_path.name}"

    return str(out_path), clip_url


def _parse_segment_datetime(segment: Path, tzinfo) -> datetime:
    day_str = segment.parent.name
    time_token = segment.stem.split(".")[0][:6]
    dt_obj = datetime.strptime(day_str + time_token, "%Y%m%d%H%M%S")
    if tzinfo is not None:
        dt_obj = dt_obj.replace(tzinfo=tzinfo)
    return dt_obj


def _concat_segments_ffmpeg(segments: Iterable[Path], out_path: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
        for seg in segments:
            tmp.write(f"file {shlex.quote(str(seg))}\n")
        list_path = tmp.name

    try:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        _run_ffmpeg(cmd)
    finally:
        os.unlink(list_path)


def _concat_segments_cv2(segments: Iterable[Path], out_path: Path) -> None:
    writer = None
    fps = 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for seg in segments:
        cap = cv2.VideoCapture(str(seg))
        if not cap.isOpened():
            continue
        seg_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if writer is None:
            fps = seg_fps
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                cap.release()
                raise RuntimeError(f"Não foi possível criar arquivo temporário: {out_path}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        cap.release()

    if writer is None:
        raise RuntimeError("Nenhum segmento pôde ser aberto para concatenação")

    writer.release()


def collect_clip(
    buffer_dir: str | Path,
    camera_id: str,
    when: datetime,
    pre: float = 5.0,
    post: float = 5.0,
    segment_seconds: float = 2.0,
) -> Tuple[str, str]:
    """Reconstrói um clipe a partir do buffer segmentado gravado via RTSP."""

    tzinfo = when.tzinfo
    buffer_root = Path(buffer_dir) / camera_id
    window_start = when - timedelta(seconds=pre)
    window_end = when + timedelta(seconds=post)

    if not buffer_root.exists():
        raise FileNotFoundError(f"Buffer da câmera '{camera_id}' não encontrado em {buffer_root}")

    segments = []
    current = window_start.date() - timedelta(days=1)
    end_date = window_end.date() + timedelta(days=1)

    while current <= end_date:
        day_dir = buffer_root / current.strftime("%Y%m%d")
        if day_dir.exists():
            for seg in sorted(day_dir.glob("*.m4s")):
                try:
                    seg_start = _parse_segment_datetime(seg, tzinfo)
                except ValueError:
                    continue
                seg_end = seg_start + timedelta(seconds=segment_seconds)
                if seg_end >= window_start and seg_start <= window_end:
                    segments.append((seg_start, seg))
        current += timedelta(days=1)

    if not segments:
        raise FileNotFoundError("Nenhum segmento encontrado para montar o clipe solicitado")

    segments.sort(key=lambda item: item[0])
    ordered_paths = [seg for _, seg in segments]
    first_start = segments[0][0]

    tmp_name = CLIPS_DIR / f"tmp_{uuid.uuid4().hex}.mp4"
    try:
        if has_ffmpeg():
            _concat_segments_ffmpeg(ordered_paths, tmp_name)
        else:
            _concat_segments_cv2(ordered_paths, tmp_name)

        event_ts = max(0.0, (when - first_start).total_seconds())
        local_path, rel = save_clip_from_file(
            src_video=str(tmp_name),
            event_ts_sec=event_ts,
            pre_sec=pre,
            post_sec=post,
            camera_id=camera_id,
            api_base_url=None,
        )
    finally:
        if tmp_name.exists():
            tmp_name.unlink()

    return local_path, rel

