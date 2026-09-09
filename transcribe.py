"""Локальная расшифровка речи из видео (mp4) -> текстовый файл.

Обрабатывает все видео из папки input (рядом со скриптом).
Расшифровки сохраняются в папку output. Обрабатываются только те видео,
для которых ещё нет соответствующего .txt файла в папке output.

Работает полностью офлайн: видео не покидает машину.
Для извлечения аудио используется встроенный ffmpeg-бинарь
из пакета imageio-ffmpeg (скачивается один раз при установке).
Модель Whisper скачивается один раз при первом запуске, далее
инференс происходит локально (без интернета).

Зависимости:
    pip install faster-whisper torch imageio-ffmpeg
"""

import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

try:
    import msvcrt
except ImportError:  # не Windows
    msvcrt = None

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm",
                    ".mpg", ".mpeg", ".wmv", ".flv", ".ts", ".m4a"}

MODEL_SIZE = "small"
LANGUAGE = None


def format_timestamp(seconds: float) -> str:
    """Переводит секунды в формат ЧЧ:ММ:СС."""
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_ffmpeg_path() -> str:
    """Возвращает путь к ffmpeg-бинарю из imageio-ffmpeg."""
    try:
        import imageio_ffmpeg
    except ImportError:
        raise RuntimeError(
            "Не установлен пакет imageio-ffmpeg. Установите:\n"
            "  pip install imageio-ffmpeg"
        )
    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_audio(video_path: Path, ffmpeg_cmd: str) -> Path:
    """Извлекает аудиодорожку из видео во временный WAV-файл.

    Возвращает путь к WAV-файлу.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Файл не найден: {video_path}")

    tmp_dir = tempfile.gettempdir()
    wav_path = Path(tmp_dir) / f"{video_path.stem}_{uuid.uuid4().hex}.wav"

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


def _cuda_available() -> bool:
    try:
        import torch  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


def load_transcriber():
    """Загружает модель Whisper один раз и возвращает функцию транскрибации.

    Используется faster-whisper, при его отсутствии — openai-whisper.
    """
    try:
        from faster_whisper import WhisperModel

        device = "cuda" if _cuda_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)

        def transcribe(audio_path: Path):
            segments, _ = model.transcribe(
                str(audio_path), language=LANGUAGE or None
            )
            return [(segment.start, segment.text.strip()) for segment in segments]

        return transcribe, "faster-whisper"
    except ImportError:
        pass

    try:
        import whisper

        device = "cuda" if _cuda_available() else "cpu"
        model = whisper.load_model(MODEL_SIZE, device=device)

        def transcribe(audio_path: Path):
            result = model.transcribe(
                str(audio_path), language=LANGUAGE or None, verbose=False
            )
            return [
                (int(seg["start"]), seg["text"].strip())
                for seg in result["segments"]
            ]

        return transcribe, "openai-whisper"
    except ImportError:
        pass

    sys.exit(
        "Не установлен ни один из бэкендов. Установите:\n"
        "  pip install faster-whisper torch\n"
        "или\n"
        "  pip install openai-whisper"
    )


def write_transcript(segments, out_path: Path) -> None:
    """Записывает реплики в формате 'ЧЧ:ММ:СС: Текст'."""
    lines = [f"{format_timestamp(start)}: {text}" for start, text in segments]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_videos(input_dir: Path) -> list[Path]:
    """Возвращает отсортированный список видео в указанной папке."""
    if not input_dir.is_dir():
        return []
    return sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def process_video(
    video_path: Path, out_path: Path, transcriber, ffmpeg_cmd: str
) -> None:
    """Расшифровывает одно видео и сохраняет результат."""
    print(f"  Извлечение аудио: {video_path.name}")
    audio_path = extract_audio(video_path, ffmpeg_cmd)
    try:
        print(f"  Транскрибация (модель {MODEL_SIZE})... это может занять время")
        segments = transcriber(audio_path)
        write_transcript(segments, out_path)
        print(f"  Сохранено: {out_path} (реплик: {len(segments)})")
    finally:
        audio_path.unlink(missing_ok=True)


def wait_for_keypress() -> None:
    """Ожидает нажатия клавиши перед закрытием окна консоли."""
    if msvcrt:
        print("\nНажмите любую клавишу, чтобы закрыть...")
        msvcrt.getch()
    else:
        input("\nНажмите Enter, чтобы закрыть...")


def main() -> None:
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    videos = find_videos(INPUT_DIR)
    if not videos:
        print(f"В папке {INPUT_DIR} нет видеофайлов.")
        wait_for_keypress()
        return

    pending = []
    for video in videos:
        out_path = OUTPUT_DIR / f"{video.stem}.txt"
        if out_path.exists():
            print(f"Пропуск (уже есть расшифровка): {video.name}")
        else:
            pending.append((video, out_path))

    if not pending:
        print("Нет видео без расшифровки, обрабатывать нечего.")
        wait_for_keypress()
        return

    print(f"Подлежит обработке: {len(pending)} из {len(videos)} видео.\n")
    transcriber, backend = load_transcriber()
    ffmpeg_cmd = get_ffmpeg_path()

    successful = []
    failed = []

    for video, out_path in pending:
        print(f"Обработка: {video.name}")
        try:
            process_video(video, out_path, transcriber, ffmpeg_cmd)
            successful.append(video.name)
        except Exception as exc:
            failed.append((video.name, str(exc)))
            print(f"  ОШИБКА: {exc}")
        print()

    print("=" * 60)
    if successful:
        print(f"Успешно обработано ({len(successful)}):")
        for name in successful:
            print(f"  [OK] {name}")
    if failed:
        print(f"Ошибки ({len(failed)}):")
        for name, error in failed:
            print(f"  [ОШИБКА] {name}: {error}")
    print("=" * 60)

    wait_for_keypress()


if __name__ == "__main__":
    main()