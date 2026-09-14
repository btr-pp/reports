"""音高追蹤：把音訊變成 NoteEvent（秒、實音 MIDI）。

兩個後端：
  pyin        - librosa 的機率式 YIN，單音（人聲、獨奏）效果好，純 pip 不需要深度學習框架。
  basic-pitch - Spotify 的多音轉譜模型，混音/器樂較適合；需要 `pip install basic-pitch`。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from .model import NoteEvent

log = logging.getLogger(__name__)

FMIN_HZ = 65.0    # C2
FMAX_HZ = 1200.0  # 約 D6


def basic_pitch_available() -> bool:
    try:
        import basic_pitch.inference  # noqa: F401

        return True
    except Exception:
        return False


# ---------------------------------------------------------------- pyin

def track_f0_pyin(y: np.ndarray, sr: int, hop: int = 256, fmin: float = FMIN_HZ, fmax: float = FMAX_HZ):
    import librosa

    f0, voiced, prob = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop, frame_length=2048, fill_na=np.nan,
    )
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)
    return times, f0, prob


def f0_to_notes(
    times: np.ndarray,
    f0: np.ndarray,
    prob: np.ndarray | None = None,
    prob_threshold: float = 0.5,
    min_duration: float = 0.07,
    median_frames: int = 5,
    hysteresis: float = 0.6,
    max_gap: float = 0.04,
) -> list[NoteEvent]:
    """把逐幀 f0 切成音符。

    - 先在 MIDI 空間做中值濾波去抖動。
    - 用 hysteresis：只要連續音高與目前音符的差距 < hysteresis 半音就視為同一音（吃掉抖音、滑音）。
    - 短於 min_duration 的音符丟掉；同音、間隔小於 max_gap 的段落合併。
    """
    if len(times) < 2:
        return []
    dt = float(np.median(np.diff(times)))
    midi = np.full(len(f0), np.nan)
    valid = np.isfinite(f0) & (f0 > 0)
    if prob is not None:
        valid &= np.nan_to_num(prob) >= prob_threshold
    midi[valid] = 69 + 12 * np.log2(f0[valid] / 440.0)
    if median_frames > 1:
        # 只在有效區段內做中值濾波（nan 會傳染，所以先填補再還原遮罩）
        filled = np.where(valid, midi, 0.0)
        smoothed = median_filter(filled, size=median_frames, mode="nearest")
        midi = np.where(valid, smoothed, np.nan)

    notes: list[NoteEvent] = []
    cur_start = None
    cur_vals: list[float] = []
    cur_conf: list[float] = []

    def close(end_idx: int):
        nonlocal cur_start, cur_vals, cur_conf
        if cur_start is not None and cur_vals:
            onset = times[cur_start]
            offset = times[end_idx - 1] + dt
            pitch = int(round(float(np.median(cur_vals))))
            conf = float(np.mean(cur_conf)) if cur_conf else 1.0
            notes.append(NoteEvent(onset, offset, pitch, conf))
        cur_start, cur_vals, cur_conf = None, [], []

    for i, m in enumerate(midi):
        if not np.isfinite(m):
            close(i)
            continue
        if cur_start is None:
            cur_start, cur_vals = i, [m]
            cur_conf = [float(prob[i])] if prob is not None else []
            continue
        center = float(np.median(cur_vals))
        if abs(m - center) < hysteresis:
            cur_vals.append(m)
            if prob is not None:
                cur_conf.append(float(prob[i]))
        else:
            close(i)
            cur_start, cur_vals = i, [m]
            cur_conf = [float(prob[i])] if prob is not None else []
    close(len(midi))

    return clean_notes(notes, min_duration=min_duration, max_gap=max_gap)


def clean_notes(notes: list[NoteEvent], min_duration: float = 0.07, max_gap: float = 0.04) -> list[NoteEvent]:
    """合併同音小間隔、移除過短音符。"""
    notes = sorted(notes, key=lambda n: n.onset)
    merged: list[NoteEvent] = []
    for n in notes:
        if merged and merged[-1].midi == n.midi and n.onset - merged[-1].offset <= max_gap:
            merged[-1].offset = max(merged[-1].offset, n.offset)
            continue
        merged.append(NoteEvent(n.onset, n.offset, n.midi, n.confidence))
    out: list[NoteEvent] = []
    for n in merged:
        if n.duration < min_duration:
            # 太短：如果前後是同一個音就併進去，否則丟掉
            if out and out[-1].midi == n.midi and n.onset - out[-1].offset <= max_gap:
                out[-1].offset = n.offset
            continue
        out.append(n)
    return out


def transcribe_pyin(y: np.ndarray, sr: int, **kw) -> list[NoteEvent]:
    times, f0, prob = track_f0_pyin(y, sr)
    return f0_to_notes(times, f0, prob, **kw)


# ---------------------------------------------------------------- basic-pitch

def transcribe_basic_pitch(wav_path: Path, min_duration: float = 0.07) -> list[NoteEvent]:
    """用 Basic Pitch 做多音轉譜，再用「天際線 + 音量」啟發式挑出單一旋律線。"""
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    _, _, events = predict(
        str(wav_path), ICASSP_2022_MODEL_PATH,
        onset_threshold=0.5, frame_threshold=0.3,
        minimum_note_length=min_duration * 1000, minimum_frequency=FMIN_HZ, maximum_frequency=FMAX_HZ,
    )
    cands = [NoteEvent(float(s), float(e), int(p), float(a)) for (s, e, p, a, *_) in events]
    return skyline(cands, min_duration=min_duration)


def skyline(cands: list[NoteEvent], min_duration: float = 0.07, pitch_weight: float = 0.004) -> list[NoteEvent]:
    """多音 → 單旋律：重疊時保留分數較高者。分數 = 音量 + 些微偏好較高音（旋律通常在上聲部）。"""
    cands = sorted(cands, key=lambda n: n.onset)
    chosen: list[NoteEvent] = []
    for n in cands:
        score = n.confidence + pitch_weight * (n.midi - 60)
        if chosen and n.onset < chosen[-1].offset:
            last = chosen[-1]
            last_score = last.confidence + pitch_weight * (last.midi - 60)
            overlap = min(last.offset, n.offset) - n.onset
            if overlap < 0.5 * n.duration and overlap < 0.5 * last.duration:
                # 只有輕微重疊：把前一個音截短
                last.offset = n.onset
                chosen.append(n)
            elif score > last_score:
                # 新音較強：前一個音截短（若剩太短就移除）
                last.offset = n.onset
                if last.duration < min_duration:
                    chosen.pop()
                chosen.append(n)
            # 否則丟掉新音
        else:
            chosen.append(n)
    return clean_notes(chosen, min_duration=min_duration)


# ---------------------------------------------------------------- 入口

def transcribe(y: np.ndarray, sr: int, wav_for_bp: Path | None, backend: str = "auto",
               stem_used: str = "mix", min_duration: float = 0.07) -> tuple[list[NoteEvent], str, list[str]]:
    """依 backend 選擇追蹤方式。auto：人聲軌用 pyin，其餘若有 basic-pitch 就用它。"""
    warnings: list[str] = []
    if backend == "auto":
        if stem_used == "vocals":
            backend = "pyin"
        elif basic_pitch_available() and wav_for_bp is not None:
            backend = "basic-pitch"
        else:
            backend = "pyin"
            if stem_used != "vocals":
                warnings.append("未安裝 basic-pitch，對混音/器樂改用單音 pyin 追蹤，準確度較低；可 `pip install 'yt2trumpet[polyphonic]'`。")
    if backend == "basic-pitch":
        if wav_for_bp is None:
            raise ValueError("basic-pitch 後端需要 wav 檔路徑")
        return transcribe_basic_pitch(wav_for_bp, min_duration=min_duration), backend, warnings
    if backend == "pyin":
        return transcribe_pyin(y, sr, min_duration=min_duration), backend, warnings
    raise ValueError(f"未知的音高後端：{backend}")
