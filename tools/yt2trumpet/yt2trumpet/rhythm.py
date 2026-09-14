"""節拍偵測與量化：秒 → 拍。"""
from __future__ import annotations

import logging
import math

import numpy as np

from .model import NoteEvent, QNote

log = logging.getLogger(__name__)


def estimate_beats(y: np.ndarray, sr: int, bpm: float | None = None, hop: int = 512) -> tuple[float, np.ndarray]:
    """回傳 (tempo, beat_times)。指定 bpm 時會強烈偏向該速度。"""
    import librosa

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    kwargs = dict(onset_envelope=onset_env, sr=sr, hop_length=hop, units="time")
    if bpm:
        kwargs.update(start_bpm=bpm, tightness=400)
    tempo, beats = librosa.beat.beat_track(**kwargs)
    tempo = float(np.atleast_1d(tempo)[0])
    beats = np.asarray(beats, dtype=float)
    if len(beats) >= 2:
        tempo = 60.0 / float(np.median(np.diff(beats)))
    return tempo, beats


def fixed_grid(duration: float, bpm: float, offset: float = 0.0) -> np.ndarray:
    """固定速度的拍點格線（使用者指定 --bpm 與 --offset 時）。"""
    period = 60.0 / bpm
    n = int(math.ceil((duration - offset) / period)) + 1
    return offset + period * np.arange(max(n, 2))


def extend_beats(beats: np.ndarray, t_min: float, t_max: float) -> np.ndarray:
    """把拍點格線往前後延伸，涵蓋所有音符。"""
    if len(beats) < 2:
        raise ValueError("拍點太少，請改用 --bpm 指定速度")
    period = float(np.median(np.diff(beats)))
    front = []
    t = beats[0] - period
    while t > t_min - period:
        front.append(t)
        t -= period
    back = []
    t = beats[-1] + period
    while t < t_max + period:
        back.append(t)
        t += period
    return np.concatenate([np.array(front[::-1]), beats, np.array(back)])


def time_to_beat(t: np.ndarray | float, beats: np.ndarray) -> np.ndarray:
    """秒 → 拍（浮點，以 beats[0] 為 0）。拍與拍之間線性內插。"""
    idx = np.arange(len(beats), dtype=float)
    return np.interp(t, beats, idx)


def estimate_downbeat_phase(starts_in_beats: np.ndarray, durs: np.ndarray, beats_per_bar: int) -> int:
    """猜第一個強拍落在哪個拍點：長音／落在整拍的音較常出現在小節開頭。"""
    if len(starts_in_beats) == 0:
        return 0
    scores = np.zeros(beats_per_bar)
    for s, d in zip(starts_in_beats, durs):
        if abs(s - round(s)) < 1e-6:
            scores[int(round(s)) % beats_per_bar] += 1.0 + d
    return int(np.argmax(scores))


def quantize(
    notes: list[NoteEvent],
    beats: np.ndarray,
    beats_per_bar: int = 4,
    grid: int = 4,
    downbeat: int | None = None,
) -> tuple[list[QNote], int]:
    """把音符起迄對齊到每拍 grid 等分（grid=4 → 十六分音符）。

    回傳 (量化音符, 第一個強拍在拍點序列中的索引)。量化音符的 start 以第一個強拍為 0，
    強拍之前的音符會被放進一個完整的弱起小節（start 會加上 beats_per_bar 的倍數）。
    """
    if not notes:
        return [], 0
    t_min = min(n.onset for n in notes)
    t_max = max(n.offset for n in notes)
    beats = extend_beats(beats, t_min, t_max)
    step = 1.0 / grid

    raw = []
    for n in notes:
        s = float(time_to_beat(n.onset, beats))
        e = float(time_to_beat(n.offset, beats))
        qs = round(s * grid) / grid
        qe = round(e * grid) / grid
        if qe <= qs:
            qe = qs + step
        raw.append([qs, qe, n.midi])

    # 去重疊：後一個音的起點切掉前一個音的尾巴
    raw.sort(key=lambda r: r[0])
    out = []
    for r in raw:
        if out and r[0] < out[-1][1]:
            out[-1][1] = r[0]
            if out[-1][1] - out[-1][0] < step - 1e-9:
                out.pop()
        if out and out[-1][0] == r[0]:
            # 同一個起點：保留較長者
            if r[1] - r[0] > out[-1][1] - out[-1][0]:
                out[-1] = r
            continue
        out.append(r)

    if downbeat is None:
        starts = np.array([r[0] for r in out])
        durs = np.array([r[1] - r[0] for r in out])
        downbeat = estimate_downbeat_phase(starts, durs, beats_per_bar)

    # 以強拍為 0；若有音符在強拍前，整體往後推整數個小節
    shift = -downbeat
    min_start = min(r[0] for r in out) + shift
    if min_start < 0:
        shift += beats_per_bar * math.ceil(-min_start / beats_per_bar)
    qnotes = [QNote(r[0] + shift, r[1] - r[0], r[2]) for r in out]
    return qnotes, downbeat


def split_long_rests(qnotes: list[QNote], max_rest_bars: int, beats_per_bar: int) -> list[QNote]:
    """把超長的空白（例如前奏、間奏）壓縮到 max_rest_bars 小節，樂譜不會一大片休止符。"""
    if not qnotes or max_rest_bars <= 0:
        return qnotes
    limit = max_rest_bars * beats_per_bar
    out = [QNote(qnotes[0].start, qnotes[0].dur, qnotes[0].midi)]
    shift = 0.0
    # 開頭前奏
    lead = qnotes[0].start
    if lead > limit:
        bars_to_cut = math.floor((lead - limit) / beats_per_bar)
        shift -= bars_to_cut * beats_per_bar
        out[0].start += shift
    for prev, n in zip(qnotes, qnotes[1:]):
        gap = n.start - prev.end
        if gap > limit:
            bars_to_cut = math.floor((gap - limit) / beats_per_bar)
            shift -= bars_to_cut * beats_per_bar
        out.append(QNote(n.start + shift, n.dur, n.midi))
    return out
