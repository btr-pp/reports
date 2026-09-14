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


def detect_onsets(y: np.ndarray, sr: int, hop: int = 256) -> np.ndarray:
    """以諧波成分（去掉鼓）的 onset strength 找起音時間點（秒）。"""
    import librosa

    y_h = librosa.effects.harmonic(y, margin=3.0)
    env = librosa.onset.onset_strength(y=y_h, sr=sr, hop_length=hop)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop, units="frames",
                                        backtrack=False, delta=0.07, wait=int(0.06 * sr / hop))
    return librosa.frames_to_time(frames, sr=sr, hop_length=hop)


def split_at_onsets(notes: list[NoteEvent], onsets: np.ndarray, y: np.ndarray, sr: int,
                    min_duration: float = 0.07, guard: float = 0.15, dip_ratio: float = 0.5,
                    hop: int = 256) -> list[NoteEvent]:
    """同一個音高的長段內若有起音、且起音前能量明顯下沉，就在那裡切成兩個音（同音反覆）。

    抖音、顫音只會讓能量小幅起伏，不會滿足 dip_ratio，所以不會被切碎。
    """
    import librosa

    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
    t_rms = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    out: list[NoteEvent] = []
    for n in notes:
        in_note = rms[(t_rms >= n.onset) & (t_rms < n.offset)]
        ref = float(np.median(in_note)) if len(in_note) else 0.0
        cuts = []
        for t in onsets:
            if not (n.onset + guard <= t <= n.offset - min_duration):
                continue
            w = rms[(t_rms >= t - 0.1) & (t_rms <= t)]
            if len(w) and ref > 0 and w.min() < dip_ratio * ref:
                cuts.append(float(t))
        start = n.onset
        for t in cuts:
            if t - start >= min_duration:
                out.append(NoteEvent(start, t, n.midi, n.confidence))
                start = t
        out.append(NoteEvent(start, n.offset, n.midi, n.confidence))
    return out


def transcribe_pyin(y: np.ndarray, sr: int, split_repeats: bool = True, **kw) -> list[NoteEvent]:
    times, f0, prob = track_f0_pyin(y, sr)
    notes = f0_to_notes(times, f0, prob, **kw)
    if split_repeats and notes:
        notes = split_at_onsets(notes, detect_onsets(y, sr), y, sr, min_duration=kw.get("min_duration", 0.07))
    return notes


# ---------------------------------------------------------------- basic-pitch

def transcribe_basic_pitch(wav_path: Path, min_duration: float = 0.07, split_repeats: bool = True) -> list[NoteEvent]:
    """用 Basic Pitch 做多音轉譜，再用「天際線 + 音量」啟發式挑出單一旋律線。"""
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    _, _, events = predict(
        str(wav_path), ICASSP_2022_MODEL_PATH,
        onset_threshold=0.5, frame_threshold=0.3,
        minimum_note_length=min_duration * 1000, minimum_frequency=FMIN_HZ, maximum_frequency=FMAX_HZ,
    )
    cands = [NoteEvent(float(s), float(e), int(p), float(a)) for (s, e, p, a, *_) in events]
    notes = select_melody(cands, min_duration=min_duration)
    if split_repeats and notes:
        import soundfile as sf

        y, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        notes = split_at_onsets(notes, detect_onsets(y, sr), y, sr, min_duration=min_duration)
    return notes


def select_melody(cands: list[NoteEvent], frame: float = 0.02, min_duration: float = 0.07,
                  w_amp: float = 1.0, w_pitch: float = 0.02, none_score: float = 0.25,
                  switch_penalty: float = 0.6, jump_penalty: float = 0.06) -> list[NoteEvent]:
    """多音 → 單旋律：以固定小時間格做 Viterbi。

    每格的狀態 = 當時正在響的某個音（或「無」）。
    得分：音量（w_amp）+ 偏好高音（w_pitch × 相對 C4 的半音數）。
    轉移：留在同一個音免費；換到「此刻剛起音」的音只付跳躍距離的小罰分；
          換到「早就在響」的音（多半是伴奏）要付 switch_penalty。
    這比單純天際線更能守住旋律線，不會被同音量的和弦音拉走。
    """
    if not cands:
        return []
    cands = sorted(cands, key=lambda n: n.onset)
    t_end = max(n.offset for n in cands)
    n_frames = int(np.ceil(t_end / frame)) + 1
    # 每格活躍的候選 index
    active: list[list[int]] = [[] for _ in range(n_frames)]
    for i, n in enumerate(cands):
        f0 = int(n.onset / frame)
        f1 = max(int(np.ceil(n.offset / frame)), f0 + 1)
        for f in range(f0, min(f1, n_frames)):
            active[f].append(i)
    NONE = -1
    emit = {i: w_amp * n.confidence + w_pitch * (n.midi - 60) for i, n in enumerate(cands)}
    onset_frame = {i: int(n.onset / frame) for i, n in enumerate(cands)}

    prev_scores: dict[int, float] = {NONE: 0.0}
    back: list[dict[int, int]] = []
    for f in range(n_frames):
        states = active[f] + [NONE]
        cur: dict[int, float] = {}
        bp: dict[int, int] = {}
        for sidx in states:
            e = none_score if sidx == NONE else emit[sidx]
            best, best_prev = -1e18, NONE
            for pidx, ps in prev_scores.items():
                if pidx == sidx:
                    trans = 0.0
                elif sidx == NONE:
                    trans = 0.0
                else:
                    jump = 0.0 if pidx == NONE else abs(cands[sidx].midi - cands[pidx].midi)
                    fresh = onset_frame[sidx] >= f - 1
                    trans = -(jump_penalty * jump) - (0.0 if fresh else switch_penalty)
                sc = ps + trans
                if sc > best:
                    best, best_prev = sc, pidx
            cur[sidx] = best + e
            bp[sidx] = best_prev
        prev_scores = cur
        back.append(bp)

    # 回溯
    state = max(prev_scores, key=prev_scores.get)
    path = [state]
    for f in range(n_frames - 1, 0, -1):
        state = back[f][state]
        path.append(state)
    path.reverse()

    # 路徑 → 音符（連續相同候選 = 一個音；同一候選被中斷後再回來算新音）
    out: list[NoteEvent] = []
    cur_idx, cur_start = NONE, 0
    for f, sidx in enumerate(path + [NONE]):
        if sidx != cur_idx:
            if cur_idx != NONE:
                c = cands[cur_idx]
                out.append(NoteEvent(max(cur_start * frame, c.onset), min(f * frame, c.offset), c.midi, c.confidence))
            cur_idx, cur_start = sidx, f
    return clean_notes([n for n in out if n.offset > n.onset], min_duration=min_duration)


def skyline(cands: list[NoteEvent], min_duration: float = 0.07, pitch_weight: float = 0.004) -> list[NoteEvent]:
    """舊版簡單天際線（保留給比較用）。"""
    cands = sorted(cands, key=lambda n: n.onset)
    chosen: list[NoteEvent] = []
    for n in cands:
        score = n.confidence + pitch_weight * (n.midi - 60)
        if chosen and n.onset < chosen[-1].offset:
            last = chosen[-1]
            last_score = last.confidence + pitch_weight * (last.midi - 60)
            overlap = min(last.offset, n.offset) - n.onset
            if overlap < 0.5 * n.duration and overlap < 0.5 * last.duration:
                last.offset = n.onset
                chosen.append(n)
            elif score > last_score:
                last.offset = n.onset
                if last.duration < min_duration:
                    chosen.pop()
                chosen.append(n)
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
            backend = "pyin"  # 單音、抖音滑音處理較好
        elif basic_pitch_available() and wav_for_bp is not None:
            backend = "basic-pitch"
        else:
            backend = "pyin"
            warnings.append("未安裝 basic-pitch，對混音/器樂改用單音 pyin 追蹤，準確度較低；建議 `pip install 'yt2trumpet[polyphonic]'`。")
    if backend == "basic-pitch":
        if wav_for_bp is None:
            raise ValueError("basic-pitch 後端需要 wav 檔路徑")
        return transcribe_basic_pitch(wav_for_bp, min_duration=min_duration), backend, warnings
    if backend == "pyin":
        return transcribe_pyin(y, sr, min_duration=min_duration), backend, warnings
    raise ValueError(f"未知的音高後端：{backend}")
