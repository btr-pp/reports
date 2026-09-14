"""從 music21 樂譜合成測試音訊（有標準答案），模擬人聲抖音、伴奏、鼓、速度漂移、殘響。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SR = 22050


@dataclass
class Scenario:
    name: str
    bpm: float = 100.0
    timbre: str = "vocal"       # vocal / piano / sine
    vibrato_cents: float = 25.0
    vibrato_hz: float = 5.5
    portamento: float = 0.04    # 相鄰音之間滑音秒數
    accomp: float = 0.0         # 和弦伴奏音量（0 = 無）
    bass: float = 0.0
    drums: float = 0.0
    drift: float = 0.0          # 速度漂移比例（0.03 = ±3%）
    reverb: float = 0.0         # 殘響濕度
    noise: float = 0.0
    lead_in_beats: float = 0.0  # 開頭空拍


def score_to_events(score):
    """回傳 (events[(start_beat, dur_beat, midi)], beats_per_bar, key)。"""
    from music21 import meter

    part = score.parts[0] if hasattr(score, "parts") and len(score.parts) else score
    flat = part.flatten()
    ts = flat.getElementsByClass(meter.TimeSignature)
    bpb = ts[0].numerator * 4 / ts[0].denominator if ts else 4
    # 弱起小節：music21 的 offset 會把不完整的第一小節算進去，這裡以第一個完整小節線為 0
    measures = part.getElementsByClass("Measure")
    shift = 0.0
    if measures and measures[0].barDuration.quarterLength != measures[0].duration.quarterLength:
        shift = measures[0].barDuration.quarterLength - measures[0].duration.quarterLength
    events = []
    for n in flat.notes:
        if n.isRest:
            continue
        p = n.pitches[-1] if n.isChord else n.pitch
        events.append((float(n.offset) + shift, float(n.quarterLength), p.midi))
    key = score.analyze("key")
    return events, int(round(bpb)), key


def beat_to_time(beats: np.ndarray, bpm: float, drift: float, rng) -> np.ndarray:
    """有漂移的拍→秒對應（拍點格線的積分）。"""
    n = int(np.ceil(beats.max())) + 8
    period = 60.0 / bpm
    if drift:
        phase = rng.uniform(0, 2 * np.pi)
        periods = period * (1 + drift * np.sin(2 * np.pi * np.arange(n) / 24 + phase))
    else:
        periods = np.full(n, period)
    grid_t = np.concatenate([[0.0], np.cumsum(periods)])
    return np.interp(beats, np.arange(n + 1), grid_t)


def _tone(f0_curve: np.ndarray, sr: int, timbre: str) -> np.ndarray:
    phase = 2 * np.pi * np.cumsum(f0_curve) / sr
    n = len(f0_curve)
    t = np.arange(n) / sr
    if timbre == "sine":
        return np.sin(phase)
    if timbre == "vocal":
        amps = [1.0, 0.7, 0.55, 0.4, 0.25, 0.18, 0.12, 0.08]
        y = sum(a * np.sin(k * phase) for k, a in enumerate(amps, start=1))
        return y / sum(amps)
    if timbre == "piano":
        amps = [1.0, 0.5, 0.3, 0.2, 0.1]
        y = sum(a * np.sin(k * phase) * np.exp(-t * (1.5 + k)) for k, a in enumerate(amps, start=1))
        return y / sum(amps)
    raise ValueError(timbre)


def synth(events, bpb: int, key, sc: Scenario, sr: int = SR, seed: int = 0):
    rng = np.random.default_rng(seed)
    starts = np.array([e[0] for e in events]) + sc.lead_in_beats
    ends = starts + np.array([e[1] for e in events])
    total_beats = float(ends.max()) + 2
    all_beats = np.arange(0, int(np.ceil(total_beats)) + 1, dtype=float)
    t_beats = beat_to_time(all_beats, sc.bpm, sc.drift, rng)
    t_start = beat_to_time(starts, sc.bpm, sc.drift, rng)
    t_end = beat_to_time(ends, sc.bpm, sc.drift, rng)
    # 注意：beat_to_time 內部用 rng 產生 phase，要用同一個 phase → 重設 rng
    rng = np.random.default_rng(seed)
    t_beats = beat_to_time(all_beats, sc.bpm, sc.drift, rng)
    rng = np.random.default_rng(seed)
    t_start = beat_to_time(starts, sc.bpm, sc.drift, rng)
    rng = np.random.default_rng(seed)
    t_end = beat_to_time(ends, sc.bpm, sc.drift, rng)

    total = int((t_beats[-1] + 1.0) * sr)
    y = np.zeros(total)

    # --- 旋律
    prev_f = None
    for i, (s, e, (_, _, midi)) in enumerate(zip(t_start, t_end, events)):
        f = 440 * 2 ** ((midi - 69) / 12)
        i0, i1 = int(s * sr), int(e * sr)
        n = max(i1 - i0, 1)
        t = np.arange(n) / sr
        f_curve = np.full(n, f)
        if sc.vibrato_cents:
            vib_env = np.clip((t - 0.15) / 0.2, 0, 1)
            f_curve = f_curve * 2 ** (sc.vibrato_cents / 1200 * vib_env * np.sin(2 * np.pi * sc.vibrato_hz * t))
        gap = s - (t_end[i - 1] if i else -1)
        if sc.portamento and prev_f is not None and gap < 0.03:
            k = int(sc.portamento * sr)
            if k < n:
                f_curve[:k] = prev_f * (f / prev_f) ** (np.arange(k) / k)
        env = np.minimum(1, t / 0.03) * np.minimum(1, (t[-1] - t + 1e-3) / 0.06)
        if sc.timbre == "vocal":
            env = env * (1 + 0.08 * np.sin(2 * np.pi * 4 * t))  # 輕微顫音（音量）
        y[i0:i0 + n] += 0.5 * env * _tone(f_curve, sr, sc.timbre)
        prev_f = f

    # --- 和弦伴奏（每小節一個三和音，以旋律音挑最合的 I/IV/V/vi/ii）
    if sc.accomp or sc.bass:
        tonic = key.tonic.midi % 12
        major = key.mode == "major"
        scale = [0, 2, 4, 5, 7, 9, 11] if major else [0, 2, 3, 5, 7, 8, 10]
        degrees = [0, 3, 4, 5, 1] if major else [0, 3, 4, 2, 5]
        nbars = int(np.ceil(total_beats / bpb))
        for bar in range(nbars):
            b0, b1 = bar * bpb, (bar + 1) * bpb
            mel = [m for (s, d, m) in events if b0 <= s + sc.lead_in_beats < b1]
            best, best_score = None, -1
            for deg in degrees:
                pcs = {(tonic + scale[(deg + k) % 7]) % 12 for k in (0, 2, 4)}
                score = sum(1 for m in mel if m % 12 in pcs)
                if score > best_score:
                    best, best_score = sorted(pcs), score
            root = (tonic + scale[degrees[0]]) % 12
            for beat in range(bpb):
                bt = b0 + beat
                if bt >= len(t_beats):
                    break
                i0 = int(t_beats[bt] * sr)
                dur = int(0.9 * (t_beats[min(bt + 1, len(t_beats) - 1)] - t_beats[bt]) * sr)
                if dur <= 0:
                    continue
                t = np.arange(dur) / sr
                if sc.accomp:
                    for pc in best:
                        midi = 48 + ((pc - 48) % 12)  # 3rd octave
                        f = 440 * 2 ** ((midi - 69) / 12)
                        y[i0:i0 + dur] += sc.accomp * _tone(np.full(dur, f), sr, "piano") * np.exp(-t * 2)
                if sc.bass and beat % 2 == 0:
                    bass_midi = 36 + ((min(best) - 36) % 12)
                    f = 440 * 2 ** ((bass_midi - 69) / 12)
                    y[i0:i0 + dur] += sc.bass * np.sin(2 * np.pi * f * t) * np.exp(-t * 3)

    # --- 鼓
    if sc.drums:
        for bt in range(len(t_beats)):
            i0 = int(t_beats[bt] * sr)
            k = 600 if bt % bpb == 0 else 300
            burst = rng.standard_normal(k) * np.exp(-np.arange(k) / (k / 5))
            if bt % bpb == 0:
                burst += 2 * np.sin(2 * np.pi * 60 * np.arange(k) / sr) * np.exp(-np.arange(k) / 200)
            y[i0:i0 + k] += sc.drums * burst[: len(y[i0:i0 + k])]

    if sc.reverb:
        ir_len = int(0.35 * sr)
        ir = rng.standard_normal(ir_len) * np.exp(-np.arange(ir_len) / (ir_len / 6))
        ir /= np.sqrt(np.sum(ir ** 2))
        wet = np.convolve(y, ir)[: len(y)]
        y = y + sc.reverb * wet
    if sc.noise:
        y += sc.noise * rng.standard_normal(len(y))
    y = 0.9 * y / (np.max(np.abs(y)) + 1e-9)
    truth = [(s + sc.lead_in_beats, d, m) for (s, d, m) in events]
    return y.astype(np.float32), truth, t_beats


SCENARIOS = {
    "clean": Scenario("clean", timbre="sine", vibrato_cents=0, portamento=0),
    "vocal": Scenario("vocal", timbre="vocal"),
    "vocal_drift": Scenario("vocal_drift", timbre="vocal", drift=0.03, reverb=0.2, noise=0.003),
    "band_mix": Scenario("band_mix", timbre="vocal", accomp=0.25, bass=0.3, drums=0.25, reverb=0.15, noise=0.003),
    "piano_solo": Scenario("piano_solo", timbre="piano", vibrato_cents=0, portamento=0, accomp=0.3, reverb=0.2),
    "lead_in": Scenario("lead_in", timbre="vocal", drums=0.3, lead_in_beats=8),
}
