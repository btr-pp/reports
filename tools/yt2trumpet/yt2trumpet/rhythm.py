"""節拍偵測與量化：秒 → 拍。"""
from __future__ import annotations

import logging
import math

import numpy as np

from .model import NoteEvent, QNote

log = logging.getLogger(__name__)


def onset_envelope(y: np.ndarray, sr: int, hop: int = 512) -> np.ndarray:
    import librosa

    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)


def track_beats(onset_env: np.ndarray, sr: int, hop: int = 512, bpm: float | None = None,
                tightness: float = 100.0) -> tuple[float, np.ndarray]:
    """librosa 動態規劃節拍追蹤。指定 bpm 時鎖定該速度附近。回傳 (tempo, beat_times)。"""
    import librosa

    kwargs = dict(onset_envelope=onset_env, sr=sr, hop_length=hop, units="time", tightness=tightness)
    if bpm:
        kwargs["bpm"] = bpm
    tempo, beats = librosa.beat.beat_track(**kwargs)
    tempo = float(np.atleast_1d(tempo)[0])
    beats = np.asarray(beats, dtype=float)
    if len(beats) >= 2:
        tempo = 60.0 / float(np.median(np.diff(beats)))
    return tempo, beats


def estimate_beats(y: np.ndarray, sr: int, bpm: float | None = None, hop: int = 512) -> tuple[float, np.ndarray]:
    """回傳 (tempo, beat_times)。指定 bpm 時會強烈偏向該速度。"""
    env = onset_envelope(y, sr, hop)
    return track_beats(env, sr, hop, bpm=bpm, tightness=400 if bpm else 100)


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


def estimate_downbeat_phase(starts_in_beats: np.ndarray, durs: np.ndarray, beats_per_bar: int,
                            audio_scores: np.ndarray | None = None, offset: int = 0) -> int:
    """猜第一個強拍落在哪個拍點。

    旋律線索：長音、樂句第一個音、落在整拍的音較常在小節開頭。
    音訊線索（可選）：鼓 / 貝斯的起音能量在強拍最重。audio_scores 以原始拍點序列的相位表示，
    offset 是量化時往前補的拍數，用來把兩種相位對齊。
    """
    if len(starts_in_beats) == 0:
        return 0
    scores = np.zeros(beats_per_bar)
    order = np.argsort(starts_in_beats)
    prev_end = -10.0
    for i in order:
        s, d = starts_in_beats[i], durs[i]
        if abs(s - round(s)) < 1e-6:
            phrase_start = (s - prev_end) >= 1.0
            scores[int(round(s)) % beats_per_bar] += 1.0 + d + (2.0 if phrase_start else 0.0)
        prev_end = s + d
    scores = scores / (scores.max() + 1e-9)
    if audio_scores is not None and len(audio_scores) == beats_per_bar:
        aligned = np.roll(audio_scores, offset)  # 原拍點 i 對應到補拍後的 i + offset
        scores = scores + 0.8 * aligned
    return int(np.argmax(scores))


def downbeat_scores_from_audio(y: np.ndarray, sr: int, beats: np.ndarray, beats_per_bar: int,
                               hop: int = 512) -> np.ndarray:
    """各拍點相位的「起音能量」總和（正規化到 0~1）。鼓 / 貝斯多半在強拍最重。"""
    import librosa

    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    t = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop)
    scores = np.zeros(beats_per_bar)
    counts = np.zeros(beats_per_bar)
    for i, b in enumerate(beats):
        w = env[(t >= b - 0.05) & (t <= b + 0.08)]
        if len(w):
            scores[i % beats_per_bar] += w.max()
            counts[i % beats_per_bar] += 1
    scores = scores / np.maximum(counts, 1)
    return scores / (scores.max() + 1e-9)


def quantize(
    notes: list[NoteEvent],
    beats: np.ndarray,
    beats_per_bar: int = 4,
    grid: int = 4,
    downbeat: int | None = None,
    audio_scores: np.ndarray | None = None,
) -> tuple[list[QNote], int]:
    """把音符起迄對齊到每拍 grid 等分（grid=4 → 十六分音符）。

    回傳 (量化音符, 第一個強拍在拍點序列中的索引)。量化音符的 start 以第一個強拍為 0，
    強拍之前的音符會被放進一個完整的弱起小節（start 會加上 beats_per_bar 的倍數）。
    """
    if not notes:
        return [], 0
    t_min = min(n.onset for n in notes)
    t_max = max(n.offset for n in notes)
    n_before = len(beats)
    first_beat = beats[0]
    beats = extend_beats(beats, t_min, t_max)
    n_front = int(np.searchsorted(beats, first_beat - 1e-6))  # 往前補了幾拍（相位要跟著平移）
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
        downbeat = estimate_downbeat_phase(starts, durs, beats_per_bar, audio_scores, offset=n_front)

    # 以強拍為 0；若有音符在強拍前，整體往後推整數個小節；開頭整小節的空白直接去掉
    shift = -downbeat
    min_start = min(r[0] for r in out) + shift
    if min_start < 0:
        shift += beats_per_bar * math.ceil(-min_start / beats_per_bar)
    elif min_start >= beats_per_bar:
        shift -= beats_per_bar * math.floor(min_start / beats_per_bar)
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


# ---------------------------------------------------------------- 以旋律起音校正拍點格線

def _grid_fit(frac: np.ndarray, weights: np.ndarray, tol: float = 0.09) -> float:
    """frac：各音符起音在拍內的位置（0~1）。落在拍點得 1 分、半拍 0.6、四分之一拍 0.25。"""
    d_beat = np.minimum(frac, 1 - frac)
    d_half = np.abs(frac - 0.5)
    d_quarter = np.minimum(np.abs(frac - 0.25), np.abs(frac - 0.75))
    score = np.where(d_beat < tol, 1.0, np.where(d_half < tol, 0.6, np.where(d_quarter < tol, 0.25, 0.0)))
    return float(np.sum(score * weights) / (np.sum(weights) + 1e-9))


def _ioi_mode(onsets: np.ndarray) -> float | None:
    """起音間隔（IOI）的眾數（秒）：在對數空間做直方圖，取最高的 bin 中位數。"""
    ioi = np.diff(np.sort(onsets))
    ioi = ioi[(ioi > 0.1) & (ioi < 4.0)]
    if len(ioi) < 4:
        return None
    logs = np.log2(ioi)
    hist, edges = np.histogram(logs, bins=np.arange(logs.min() - 0.05, logs.max() + 0.15, 0.1))
    k = int(np.argmax(hist))
    sel = ioi[(logs >= edges[k]) & (logs < edges[k + 1])]
    return float(np.median(sel)) if len(sel) else None


def _bpm_prior(bpm: float, center: float = 110.0, sigma_octaves: float = 1.5) -> float:
    return float(np.exp(-0.5 * (np.log2(bpm / center) / sigma_octaves) ** 2))


def refine_beats(beats: np.ndarray, notes: list[NoteEvent], onset_env: np.ndarray | None = None, sr: int = 22050,
                 hop: int = 512, bpm_min: float = 55.0, bpm_max: float = 190.0,
                 factors=(1.0, 2.0, 0.5, 1.5, 2.0 / 3.0, 4.0 / 3.0, 0.75, 3.0, 1.0 / 3.0),
                 phases: int = 8, tightness: float = 800.0, snap_eval: bool = True,
                 spread: tuple[float, ...] = (0.97, 1.0, 1.03)) -> tuple[np.ndarray, dict]:
    """用旋律起音校正節拍器的速度與相位。

    速度候選來自兩處：節拍器速度 × 常見倍率，以及旋律起音間隔眾數 × 常見倍率
    （全是二分音符的曲子節拍器常常抓出亂數，這時後者才救得回來）。
    每個候選都用同一條 onset envelope 重新追蹤拍點（有 envelope 時），再在 8 個相位中挑
    讓最多音符落在拍點 / 半拍上的組合；以溫和的「常見速度」先驗打破平手。
    回傳 (新的拍點時間, 診斷資訊)。
    """
    if len(beats) < 4 or not notes:
        return beats, {}
    onsets = np.array([n.onset for n in notes])
    weights = np.sqrt(np.array([max(n.duration, 0.05) for n in notes]))
    t_min = min(onsets.min(), beats[0]) - 2.0
    t_max = max(max(n.offset for n in notes), beats[-1]) + 2.0
    base_period = float(np.median(np.diff(beats)))
    base_bpm = 60.0 / base_period

    cands: list[tuple[float, str]] = []
    for f in factors:
        cands.append((base_bpm * f, f"tracker×{f:.2f}"))
    mode = _ioi_mode(onsets)
    if mode:
        for m in (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0, 1.5, 2.0 / 3.0):
            cands.append((60.0 / (mode * m), f"ioi×{m:.2f}"))
    cands = [(bpm * sp, src) for bpm, src in cands for sp in spread]
    # 去重（2% 內視為相同）、限制範圍
    uniq: list[tuple[float, str]] = []
    for bpm, src in sorted(cands):
        if not (bpm_min <= bpm <= bpm_max):
            continue
        if uniq and abs(bpm / uniq[-1][0] - 1) < 0.02:
            continue
        uniq.append((bpm, src))

    best = None
    for bpm, src in uniq:
        if onset_env is None:
            f = bpm / base_bpm
            grid = extend_beats(beats, t_min, t_max)
            coord = f * np.arange(len(grid), dtype=float)
        else:
            _, cand = track_beats(onset_env, sr, hop, bpm=bpm, tightness=tightness)
            if len(cand) < 4:
                continue
            if snap_eval:
                cand = snap_grid_to_onsets(cand, notes)
            grid = extend_beats(cand, t_min, t_max)
            coord = np.arange(len(grid), dtype=float)
        beta = np.interp(onsets, grid, coord)
        tol = min(0.09, 0.06 * bpm / 60.0)  # 容差以秒為上限，慢速度不會因為視窗變寬而占便宜
        prior = _bpm_prior(bpm)
        if src == "tracker×1.00":
            prior *= 1.03  # 節拍器原本的答案略優先
        for k in range(phases):
            phi = k / phases
            fit = _grid_fit(np.mod(beta - phi, 1.0), weights, tol)
            score = fit * prior * (1.0 - 0.06 * min(phi, 1 - phi) * 2)
            if best is None or score > best[0]:
                best = (score, bpm, phi, fit, grid, coord, src)
    if best is None:
        return beats, {}
    _, bpm, phi, fit, grid, coord, src = best
    new_coord = coord - phi
    n0 = int(np.ceil(new_coord[0]))
    n1 = int(np.floor(new_coord[-1]))
    new_beats = np.interp(np.arange(n0, n1 + 1, dtype=float), new_coord, grid)
    bpm = 60.0 / float(np.median(np.diff(new_beats)))
    info = {"factor": bpm / base_bpm, "phase": phi, "fit": fit, "bpm": bpm, "base_bpm": base_bpm, "source": src}
    return new_beats, info


def snap_grid_to_onsets(beats: np.ndarray, notes: list[NoteEvent], tol: float = 0.12) -> np.ndarray:
    """把拍點鎖到附近的旋律起音上（容差 tol 拍），沒有起音的拍點在鎖定的鄰居之間線性內插。

    用途：速度倍率已經選對之後，讓格線跟著演奏的速度漂移走，避免長曲子累積誤差而錯拍。
    """
    if len(beats) < 3 or not notes:
        return beats
    onsets = np.sort(np.array([n.onset for n in notes]))
    period = float(np.median(np.diff(beats)))
    snapped = beats.astype(float).copy()
    locked = np.zeros(len(beats), dtype=bool)
    for i, b in enumerate(beats):
        j = np.searchsorted(onsets, b)
        cands = [onsets[k] for k in (j - 1, j) if 0 <= k < len(onsets)]
        if not cands:
            continue
        c = min(cands, key=lambda t: abs(t - b))
        if abs(c - b) <= tol * period:
            snapped[i] = c
            locked[i] = True
    if locked.sum() < 2:
        return beats
    idx = np.arange(len(beats), dtype=float)
    li = idx[locked]
    out = np.interp(idx, li, snapped[locked])
    # 頭尾沒有鎖定的拍點：沿用最近鎖定段的平均週期外推
    first, last = int(li[0]), int(li[-1])
    if first > 0:
        out[:first] = snapped[first] - period * (first - idx[:first])
    if last < len(beats) - 1:
        out[last + 1:] = snapped[last] + period * (idx[last + 1:] - last)
    return out
