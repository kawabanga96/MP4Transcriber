"""Локальная расшифровка речи из видео (mp4) -> текстовый файл.

Работает полностью офлайн: видео не покидает машину.
Для извлечения аудио требуется ffmpeg.
Модель Whisper скачивается один раз при первом запуске, далее
инференс происходит локально (без интернета).

Зависимости:
    pip install faster-whisper torch openai-whisper
    (ffmpeg должен быть в PATH, либо задан --ffmpeg)
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def format_timestamp(seconds: float) -> str:
    """Переводит секунды в формат [ЧЧ:ММ:СС]."""
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"


def extract_audio(video_path: Path, ffmpeg_cmd: str) -> Path:
    """Извлекает аудиодорожку из видео во временный WAV-файл.

    Возвращает путь к WAV-файлу.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Файл не найден: {video_path}")

    tmp_dir = tempfile.gettempdir()
    wav_path = Path(tmp_dir) / f"{video_path.stem}_{id(video_path)}.wav"

    cmd = [
        ffmpeg_cmd, "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Ошибка ffmpeg при извлечении аудио:\n{result.stderr}"
        )
    return wav_path


def transcribe_faster_whisper(audio_path: Path, model_size: str, language: str):
    """Транскрибация через faster-whisper (предпочтительно)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None

    device = "cuda" if _cuda_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _ = model.transcribe(str(audio_path), language=language or None)
    return [(segment.start, segment.text.strip()) for segment in segments]


def transcribe_openai_whisper(audio_path: Path, model_size: str, language: str):
    """Транскрибация через openai-whisper (резервный вариант)."""
    try:
        import whisper
    except ImportError:
        return None

    device = "cuda" if _cuda_available() else "cpu"
    model = whisper.load_model(model_size, device=device)
    result = model.transcribe(str(audio_path), language=language or None, verbose=False)
    return [(int(seg["start"]), seg["text"].strip()) for seg in result["segments"]]


def _cuda_available() -> bool:
    try:
        import torch  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


def transcribe(audio_path: Path, model_size: str, language: str):
    """Пробует faster-whisper, затем openai-whisper."""
    backend = transcribe_faster_whisper(audio_path, model_size, language)
    if backend is None:
        backend = transcribe_openai_whisper(audio_path, model_size, language)
    if backend is None:
        sys.exit(
            "Не установлен ни один из бэкендов. Установите:\n"
            "  pip install faster-whisper torch\n"
            "или\n"
            "  pip install openai-whisper"
        )
    return backend


def write_transcript(segments, out_path: Path) -> None:
    """Записывает реплики в формате '[время] Текст'."""
    lines = [f"{format_timestamp(start)} {text}" for start, text in segments]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Локальная расшифровка речи из видео mp4 в текстовый файл."
    )
    parser.add_argument("video", type=Path, help="Путь к .mp4 (или иному видео) файлу")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Путь к выходному .txt файлу (по умолчанию рядом с видео)")
    parser.add_argument("-m", "--model", type=str, default="small",
                        help="Размер модели Whisper: tiny/base/small/medium/large-v3 "
                             "(по умолчанию small)")
    parser.add_argument("-l", "--language", type=str, default=None,
                        help="Код языка (например, 'ru'). Пусто = автоопределение")
    parser.add_argument("--ffmpeg", type=str, default="ffmpeg",
                        help="Путь к исполняемому файлу ffmpeg (по умолчанию 'ffmpeg')")
    args = parser.parse_args()

    out_path = args.output or args.video.with_suffix(".txt")

    print(f"Извлечение аудио из: {args.video}")
    audio_path = extract_audio(args.video, args.ffmpeg)
    try:
        print(f"Транскрибация (модель {args.model})... это может занять время")
        segments = transcribe(audio_path, args.model, args.language)
        write_transcript(segments, out_path)
        print(f"Готово! Расшифровка сохранена: {out_path}")
        print(f"Реплик: {len(segments)}")
    finally:
        audio_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()