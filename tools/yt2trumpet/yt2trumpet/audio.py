"""音訊取得：YouTube 下載、格式轉換、裁切。"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

URL_RE = re.compile(r"^(https?://|www\.)", re.IGNORECASE)


def is_url(s: str) -> bool:
    return bool(URL_RE.match(s.strip()))


def find_ffmpeg() -> str:
    """優先用系統的 ffmpeg，找不到就用 imageio-ffmpeg 內附的執行檔。"""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover - 只有兩個都缺才會到這
        raise RuntimeError("找不到 ffmpeg，請安裝 ffmpeg 或 `pip install imageio-ffmpeg`") from e


def safe_filename(name: str, max_len: int = 80) -> str:
    name = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", name).strip(" ._")
    return (name or "untitled")[:max_len]


def download_audio(url: str, work_dir: Path) -> tuple[Path, str]:
    """用 yt-dlp 下載最佳音訊並轉成 wav。回傳 (wav 路徑, 影片標題)。"""
    import yt_dlp

    work_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(work_dir / "source.%(ext)s"),
        "ffmpeg_location": str(Path(ffmpeg).parent),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    title = info.get("title") or "untitled"
    wav = work_dir / "source.wav"
    if not wav.exists():  # pragma: no cover - yt-dlp 正常都會產生 source.wav
        cands = list(work_dir.glob("source.*"))
        if not cands:
            raise RuntimeError("yt-dlp 沒有產生音訊檔")
        wav = to_wav(cands[0], work_dir / "source.wav")
    return wav, title


def to_wav(src: Path, dst: Path, sr: int | None = None, mono: bool = False,
           start: float | None = None, end: float | None = None) -> Path:
    """以 ffmpeg 轉成 PCM wav，可同時裁切、重取樣、混成單聲道。"""
    cmd = [find_ffmpeg(), "-y", "-loglevel", "error"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if end:
        dur = end - (start or 0.0)
        cmd += ["-t", f"{dur:.3f}"]
    if sr:
        cmd += ["-ar", str(sr)]
    if mono:
        cmd += ["-ac", "1"]
    cmd += ["-acodec", "pcm_s16le", str(dst)]
    subprocess.run(cmd, check=True)
    return dst


def load_mono(path: Path, sr: int = 22050) -> tuple[np.ndarray, int]:
    """讀成單聲道 float32。非 wav 或取樣率不同時先經 ffmpeg 轉檔。"""
    try:
        y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        tmp = path.with_suffix(".tmp.wav")
        to_wav(path, tmp, sr=sr, mono=True)
        y, file_sr = sf.read(str(tmp), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if file_sr != sr:
        import librosa

        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y.astype(np.float32), sr
