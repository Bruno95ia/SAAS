"""Gerenciador unificado de captura de vídeo.

A classe :class:`CaptureManager` encapsula a lógica de construção e execução

de comandos FFmpeg para diferentes tipos de origem: streams RTSP ou captura
local via AVFoundation em macOS. Também cuida de reconectar automaticamente em
caso de falhas, registrando logs detalhados.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path
from typing import List

from saas import config
from saas.utils.logger import get_logger


class CaptureManager:
    """Implementa captura contínua com reconexão automática."""

    def __init__(
        self,
        camera_id: str,
        source: str,
        output_dir: Path,
        segment_seconds: int = 2,
        reconnect_seconds: float = 5.0,
        ffmpeg_binary: str = "ffmpeg",
        ffprobe_binary: str = "ffprobe",
    ) -> None:
        self.camera_id = camera_id
        self.source = source
        self.source_type = config.detect_source_type(source)
        self.output_dir = output_dir
        self.segment_seconds = max(1, segment_seconds)
        self.reconnect_seconds = max(1.0, reconnect_seconds)
        self.ffmpeg_binary = ffmpeg_binary
        self.ffprobe_binary = ffprobe_binary
        self.logger = get_logger(f"saas.capture.{camera_id}")

        config.ensure_runtime_directories()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._segment_template = config.segment_template(self.output_dir)

        self.logger.info(
            "Inicializando captura camera=%s origem=%s tipo=%s destino=%s",
            camera_id,
            source,
            self.source_type,
            self.output_dir,
        )
        self._log_environment()
        self._probe_source()

    # ------------------------------------------------------------------ utils
    @property
    def segment_template(self) -> Path:
        """Template strftime usado para salvar os segmentos gravados."""

        return self._segment_template

    def _log_environment(self) -> None:
        system = platform.system()
        release = platform.release()
        self.logger.info("Sistema operacional: %s %s", system, release)

    def _probe_source(self) -> None:
        """Obtém informações da stream usando ffprobe (quando disponível)."""

        if self.source_type not in {"rtsp", "custom"}:
            # Para captura local/screen os parâmetros são definidos manualmente.
            self.logger.info(
                "Captura local/avfoundation detectada. Ajuste devices conforme necessário."
            )
            return

        cmd = [
            self.ffprobe_binary,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate",
            "-of",
            "json",
            self.source,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            self.logger.warning("Não foi possível executar ffprobe: %s", exc)
            return

        try:
            payload = json.loads(result.stdout)
            stream = payload["streams"][0]
            codec = stream.get("codec_name", "?")
            width = stream.get("width", "?")
            height = stream.get("height", "?")
            fps_raw = stream.get("avg_frame_rate", "0/1")
            fps = self._parse_fps(fps_raw)
            self.logger.info(
                "Stream detectada codec=%s res=%sx%s fps=%.2f", codec, width, height, fps
            )
        except Exception as exc:  # pragma: no cover - parsing defensivo
            self.logger.warning("Falha ao interpretar saída do ffprobe: %s", exc)

    @staticmethod
    def _parse_fps(expr: str) -> float:
        try:
            num, den = expr.split("/")
            num_f = float(num)
            den_f = float(den)
            return num_f / den_f if den_f else 0.0
        except Exception:
            return 0.0

    def _ffmpeg_command(self) -> List[str]:
        """Monta o comando FFmpeg apropriado para o tipo de origem."""

        template = str(self.segment_template)
        base_cmd = [
            self.ffmpeg_binary,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "info",
        ]

        if self.source_type == "rtsp":
            base_cmd += [
                "-rtsp_transport",
                "tcp",
                "-timeout",
                "5000000",
                "-i",
                self.source,
                "-fflags",
                "+genpts",
                "-reset_timestamps",
                "1",
                "-c",
                "copy",
            ]
        elif self.source_type in {"screen", "local"}:
            # Captura via AVFoundation (macOS). Ajuste dos dispositivos padrão:
            device = "1:none" if self.source_type == "screen" else "0:none"
            base_cmd += [
                "-f",
                "avfoundation",
                "-framerate",
                "30",
                "-video_size",
                "1280x720",
                "-i",
                device,
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
            ]
        else:
            # Origens customizadas: delegamos a interpretação diretamente ao FFmpeg.
            base_cmd += [
                "-i",
                self.source,
                "-fflags",
                "+genpts",
                "-reset_timestamps",
                "1",
                "-c",
                "copy",
            ]

        base_cmd += [
            "-f",
            "segment",
            "-segment_time",
            str(self.segment_seconds),
            "-strftime",
            "1",
            template,
        ]
        return base_cmd

    def _log_ffmpeg_line(self, line: str) -> None:
        text = line.strip()
        if not text:
            return
        lowered = text.lower()
        if "error" in lowered:
            self.logger.error("FFmpeg: %s", text)
        elif "warning" in lowered:
            self.logger.warning("FFmpeg: %s", text)
        else:
            self.logger.debug("FFmpeg: %s", text)

    # ----------------------------------------------------------------- public
    def run(self) -> None:
        """Executa a captura em um loop infinito com reconexão automática."""

        command = self._ffmpeg_command()
        self.logger.info("Comando FFmpeg: %s", " ".join(command))

        while True:
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "FFmpeg não encontrado. Instale ffmpeg 6.1.1+ para continuar."
                ) from exc

            start = time.time()
            self.logger.info("Captura iniciada (PID=%s)", process.pid)

            try:
                assert process.stdout is not None
                for line in process.stdout:
                    self._log_ffmpeg_line(line)
            except KeyboardInterrupt:
                self.logger.info("Interrompido pelo usuário. Encerrando captura...")
                process.terminate()
                process.wait(timeout=5)
                break

            exit_code = process.wait()
            duration = time.time() - start
            if exit_code == 0:
                self.logger.info(
                    "FFmpeg finalizado normalmente após %.1fs. Reiniciando...", duration
                )
            else:
                self.logger.warning(
                    "FFmpeg encerrou com código %s após %.1fs. Reconnect em %.1fs.",
                    exit_code,
                    duration,
                    self.reconnect_seconds,
                )
                time.sleep(self.reconnect_seconds)

    # ---------------------------------------------------------------- factory
    @classmethod
    def from_args(
        cls,
        camera_id: str,
        source: str,
        output_base: Path,
        segment_seconds: int,
        reconnect_seconds: float = 5.0,
    ) -> "CaptureManager":
        """Cria o gerenciador a partir dos argumentos da CLI."""

        output_dir = output_base / camera_id
        output_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            camera_id=camera_id,
            source=source,
            output_dir=output_dir,
            segment_seconds=segment_seconds,
            reconnect_seconds=reconnect_seconds,
        )
