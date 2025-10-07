#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess, sys, os, shlex
from pathlib import Path

def has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        return False

def convert_ffmpeg(src: Path, dst: Path, crf: int = 23, preset: str = "fast", copy_audio: bool = False) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # -c:v libx264 para compatibilidade ampla; -crf controla qualidade; -preset ajusta velocidade
    # Áudio: se copy_audio True, tenta copiar áudio; senão converte para AAC (compat).
    acmd = "-c:a copy" if copy_audio else "-c:a aac -b:a 128k"
    cmd = f'ffmpeg -y -i {shlex.quote(str(src))} -c:v libx264 -preset {preset} -crf {crf} {acmd} -movflags +faststart {shlex.quote(str(dst))}'
    return subprocess.run(cmd, shell=True).returncode

def convert_moviepy(src: Path, dst: Path, crf: int = 23, preset: str = "fast") -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        from moviepy.editor import VideoFileClip
# MoviePy não usa crf/preset diretamente; usamos defaults bons (libx264 + aac)
        with VideoFileClip(str(src)) as clip:
            clip.write_videofile(
                str(dst),
                codec="libx264",
                audio_codec="aac",
                preset=preset,
                bitrate=None,  # deixa o encoder decidir
                threads=os.cpu_count() or 1,
                verbose=False,
                logger=None,
            )
        return 0
    except Exception as e:
        print(f"[MoviePy] Falhou em {src.name}: {e}", file=sys.stderr)
        return 1

def main():
    p = argparse.ArgumentParser(description="Converter .avi para .mp4 em lote.")
    p.add_argument("-i", "--input", type=Path, default=Path("."), help="Pasta de entrada (procura recursivo por .avi).")
    p.add_argument("-o", "--output", type=Path, default=None, help="Pasta de saída (espelho da estrutura). Padrão: mesma da entrada.")
    p.add_argument("--crf", type=int, default=23, help="Qualidade (18 melhor/maior, 23 balanceado).")
    p.add_argument("--preset", type=str, default="fast", help="x264 preset (ultrafast..placebo). Padrão: fast.")
    p.add_argument("--copy-audio", action="store_true", help="Tenta copiar áudio original em vez de recomprimir.")
    p.add_argument("--overwrite", action="store_true", help="Sobrescreve .mp4 existentes.")
    p.add_argument("--dest-clips", type=Path, default=None, help="Se definido, salva todos .mp4 em uma pasta única (ex.: runs/clips).")
    args = p.parse_args()

    use_ffmpeg = has_ffmpeg()
    if not use_ffmpeg:
        print("[info] FFmpeg não encontrado. Usando MoviePy (mais lento). Instale FFmpeg para melhor performance.", file=sys.stderr)

    avi_files = [p for p in args.input.rglob("*.avi")]
    if not avi_files:
        print("[info] Nenhum .avi encontrado.")
        return 0

    ok = fail = 0
    for src in avi_files:
        if args.dest_clips:
            dst_dir = args.dest_clips
            rel_name = src.stem  # sem subpastas
        else:
            dst_dir = (args.output or args.input) / src.relative_to(args.input).parent
            rel_name = src.stem

        dst = dst_dir / f"{rel_name}.mp4"
        if dst.exists() and not args.overwrite:
            print(f"[skip] já existe: {dst}")
            ok += 1
            continue

        print(f"[conv] {src} -> {dst}")
        rc = (convert_ffmpeg(src, dst, args.crf, args.preset, args.copy_audio)
              if use_ffmpeg else convert_moviepy(src, dst, args.crf, args.preset))
        if rc == 0:
            ok += 1
        else:
            fail += 1

    print(f"[done] sucesso={ok} falhas={fail} total={len(avi_files)}")
    return 0 if fail == 0 else 1

if __name__ == "__main__":
    sys.exit(main())