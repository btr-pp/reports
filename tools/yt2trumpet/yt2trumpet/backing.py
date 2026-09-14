"""伴唱 / 伴奏音樂：把主旋律那一軌從混音裡拿掉，需要時跟著樂譜一起移調。"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import find_ffmpeg

log = logging.getLogger(__name__)


def _read_stereo(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return y, sr


def backing_from_stems(stems: dict[str, Path], melody_stem: str) -> tuple[np.ndarray, int]:
    """把 Demucs 四軌中除了旋律來源軌以外的軌加起來（例如去掉 vocals → 伴唱帶）。"""
    remove = {"vocals"} if melody_stem == "vocals" else {"other", "vocals"} if melody_stem == "other" else set()
    ys, sr = [], None
    for name, p in stems.items():
        if name in remove:
            continue
        y, sr = _read_stereo(p)
        ys.append(y)
    if not ys:
        raise ValueError("沒有可用的伴奏軌")
    n = min(len(y) for y in ys)
    return np.sum([y[:n] for y in ys], axis=0), sr


def center_cancel(wav: Path) -> tuple[np.ndarray, int]:
    """沒有 Demucs 時的陽春做法：左右聲道相減，消掉混在正中央的人聲（會一起消掉中央的貝斯、鼓）。"""
    y, sr = _read_stereo(wav)
    if y.shape[1] < 2:
        raise ValueError("單聲道音檔無法用左右相消去人聲，請安裝 demucs")
    side = (y[:, 0] - y[:, 1]) * 0.5
    return np.stack([side, side], axis=1), sr


def pitch_shift_stereo(y: np.ndarray, sr: int, semitones: int) -> np.ndarray:
    import librosa

    return np.stack([librosa.effects.pitch_shift(y[:, c], sr=sr, n_steps=semitones) for c in range(y.shape[1])], axis=1)


def normalize(y: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(y))) or 1.0
    return (y * (peak / m)).astype(np.float32) if m > peak else y.astype(np.float32)


def export(y: np.ndarray, sr: int, out_base: Path, fmt: str = "mp3") -> Path:
    """先寫 wav，需要 mp3 時用 ffmpeg 轉；轉檔失敗就留 wav。"""
    wav = out_base.with_suffix(".wav")
    sf.write(str(wav), y, sr)
    if fmt != "mp3":
        return wav
    mp3 = out_base.with_suffix(".mp3")
    try:
        subprocess.run([find_ffmpeg(), "-y", "-loglevel", "error", "-i", str(wav), "-codec:a", "libmp3lame",
                        "-q:a", "2", str(mp3)], check=True)
        wav.unlink()
        return mp3
    except Exception as e:  # pragma: no cover - 取決於 ffmpeg 編譯選項
        log.warning(f"mp3 轉檔失敗（{e}），保留 wav")
        return wav


def make_backing(stems: dict[str, Path] | None, melody_stem: str, source_wav: Path, out_base: Path,
                 transpose: int = 0, fmt: str = "mp3") -> tuple[Path, list[str]]:
    """產生伴唱 / 伴奏音檔。回傳 (檔案路徑, 警告)。"""
    warnings: list[str] = []
    if stems:
        y, sr = backing_from_stems(stems, melody_stem)
        if melody_stem == "mix":
            warnings.append("旋律來源是原始混音，伴奏只去掉了人聲軌。")
    else:
        y, sr = center_cancel(source_wav)
        warnings.append("沒有分離軌（未安裝 demucs 或選了不分離），伴奏用左右聲道相消做的，品質有限；"
                        "安裝 demucs 並選擇分離後會好很多。")
    if transpose:
        log.info(f"伴奏跟著移調 {transpose:+d} 半音…")
        y = pitch_shift_stereo(y, sr, transpose)
    return export(normalize(y), sr, out_base, fmt), warnings
