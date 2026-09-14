"""人聲 / 伴奏分離（Demucs），以及「主旋律在哪一軌」的自動判斷。"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .audio import load_mono

log = logging.getLogger(__name__)

STEMS = ("vocals", "drums", "bass", "other")
VOCAL_RATIO_THRESHOLD = 0.18  # 人聲能量占比高於此值就當成「有唱的歌」


def demucs_available() -> bool:
    try:
        import demucs  # noqa: F401
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def run_demucs(wav: Path, out_dir: Path, model: str = "htdemucs", device: str | None = None) -> dict[str, Path]:
    """以子行程呼叫 demucs，回傳 {stem 名稱: wav 路徑}。"""
    cmd = [sys.executable, "-m", "demucs", "-n", model, "-o", str(out_dir)]
    if device:
        cmd += ["-d", device]
    cmd.append(str(wav))
    log.info("執行 Demucs 分離（第一次會下載模型，CPU 需要幾分鐘）…")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Demucs 執行失敗。第一次執行需要網路下載模型（約 80 MB）；"
            "若不想分離可加 --stem none。原始錯誤請看上方輸出。") from e
    stem_dir = out_dir / model / wav.stem
    stems = {s: stem_dir / f"{s}.wav" for s in STEMS}
    missing = [s for s, p in stems.items() if not p.exists()]
    if missing:
        raise RuntimeError(f"Demucs 沒有輸出這些軌：{missing}")
    return stems


def rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y)))) if len(y) else 0.0


def vocal_ratio(stems: dict[str, Path], sr: int = 22050) -> float:
    """人聲 RMS 占四軌 RMS 總和的比例。"""
    energies = {s: rms(load_mono(p, sr)[0]) for s, p in stems.items()}
    total = sum(energies.values()) or 1e-9
    ratio = energies["vocals"] / total
    log.info("各軌能量占比：" + ", ".join(f"{s}={e / total:.2f}" for s, e in energies.items()))
    return ratio


def mix_stems(paths: list[Path], sr: int = 22050) -> np.ndarray:
    ys = [load_mono(p, sr)[0] for p in paths]
    n = min(len(y) for y in ys)
    return np.sum([y[:n] for y in ys], axis=0).astype(np.float32)


def select_melody_audio(
    wav: Path,
    work_dir: Path,
    mode: str = "auto",
    sr: int = 22050,
    model: str = "htdemucs",
    device: str | None = None,
) -> tuple[np.ndarray, str, list[str], dict[str, Path] | None]:
    """依 mode 決定拿哪一軌來追旋律。

    mode:
      auto   - 跑 Demucs，人聲占比夠高就用 vocals，否則用 other（旋律樂器多半在這）。
      vocals - 強制用人聲軌。
      other  - 強制用 other 軌（鋼琴、合成器、吉他、弦樂…）。
      none   - 不分離，直接用原始混音。
    回傳 (單聲道音訊, 實際使用的軌名, 警告列表, Demucs 四軌路徑或 None)。
    """
    warnings: list[str] = []
    if mode == "none":
        y, _ = load_mono(wav, sr)
        return y, "mix", warnings, None

    if not demucs_available():
        warnings.append("未安裝 demucs/torch，改用原始混音追旋律；流行歌建議 `pip install 'yt2trumpet[separate]'`。")
        y, _ = load_mono(wav, sr)
        return y, "mix", warnings, None

    stems = run_demucs(wav, work_dir / "stems", model=model, device=device)
    if mode == "auto":
        ratio = vocal_ratio(stems, sr)
        if ratio >= VOCAL_RATIO_THRESHOLD:
            mode = "vocals"
            log.info(f"人聲占比 {ratio:.2f} ≥ {VOCAL_RATIO_THRESHOLD}，判斷為有人聲的歌 → 用 vocals 軌")
        else:
            mode = "other"
            log.info(f"人聲占比 {ratio:.2f} < {VOCAL_RATIO_THRESHOLD}，判斷為純音樂 → 用 other 軌")

    if mode == "vocals":
        y, _ = load_mono(stems["vocals"], sr)
        return y, "vocals", warnings, stems
    if mode == "other":
        # other + vocals：純音樂裡偶爾有少量人聲/合唱也一併保留
        y = mix_stems([stems["other"], stems["vocals"]], sr)
        return y, "other", warnings, stems
    raise ValueError(f"未知的 stem 模式：{mode}")
