"""小號相關：實音→記譜音、音域折疊、調性偵測、移調決策。"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .model import KeyInfo, QNote

log = logging.getLogger(__name__)

# Bb 小號：記譜音 = 實音 + 大二度
CONCERT_TO_WRITTEN = 2

# 記譜音域（MIDI）：G3=55, C4=60, F#3=54, G5=79, C6=84, E6=88
RANGE_PRESETS: dict[str, tuple[int, int]] = {
    "beginner": (60, 79),
    "intermediate": (55, 84),
    "advanced": (54, 88),
}

_NOTE_NAMES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def parse_note_name(name: str) -> int:
    """'G3' / 'Bb4' / 'F#5' → MIDI（C4 = 60）。"""
    s = name.strip()
    letter = s[0].upper()
    rest = s[1:]
    acc = 0
    while rest and rest[0] in "#b":
        acc += 1 if rest[0] == "#" else -1
        rest = rest[1:]
    octave = int(rest)
    return 12 * (octave + 1) + _NOTE_NAMES[letter] + acc


def midi_to_name(midi: int, flats: bool = False) -> str:
    names = KeyInfo.NAMES_FLAT if flats else KeyInfo.NAMES_SHARP
    return f"{names[midi % 12]}{midi // 12 - 1}"


def parse_range(spec: str) -> tuple[int, int]:
    """'intermediate' 或 'G3-C6' → (lo, hi) 記譜 MIDI。"""
    if spec in RANGE_PRESETS:
        return RANGE_PRESETS[spec]
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return parse_note_name(lo), parse_note_name(hi)
    raise ValueError(f"看不懂的音域：{spec}（可用 beginner / intermediate / advanced 或 G3-C6）")


# ---------------------------------------------------------------- 調性

_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def key_from_chroma(chroma: np.ndarray) -> KeyInfo:
    """Krumhansl-Schmuckler：與 24 個調的 profile 做相關，取最高者。"""
    best = (-2.0, 0, "major")
    for tonic in range(12):
        for mode, prof in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
            r = float(np.corrcoef(np.roll(prof, tonic), chroma)[0, 1])
            if r > best[0]:
                best = (r, tonic, mode)
    return KeyInfo(best[1], best[2])


def detect_key_from_audio(y: np.ndarray, sr: int) -> KeyInfo:
    import librosa

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    return key_from_chroma(chroma.mean(axis=1))


def detect_key_from_notes(notes: list[QNote]) -> KeyInfo:
    """只用旋律音符（以時值加權）估調性；當音訊沒有和聲資訊時的備援。"""
    chroma = np.zeros(12)
    for n in notes:
        chroma[n.midi % 12] += n.dur
    if chroma.sum() == 0:
        return KeyInfo(0, "major")
    return key_from_chroma(chroma)


# ---------------------------------------------------------------- 音域折疊

@dataclass
class FoldStats:
    phrases: int = 0
    phrases_shifted: int = 0
    notes_shifted_individually: int = 0
    notes_clipped: int = 0


def split_phrases(notes: list[QNote], min_gap_beats: float = 1.0) -> list[list[QNote]]:
    """以 ≥ min_gap_beats 的空白切樂句。"""
    phrases: list[list[QNote]] = []
    cur: list[QNote] = []
    for n in notes:
        if cur and n.start - cur[-1].end >= min_gap_beats:
            phrases.append(cur)
            cur = []
        cur.append(n)
    if cur:
        phrases.append(cur)
    return phrases


def _out_of_range(midis, lo, hi) -> int:
    return sum(1 for m in midis if m < lo or m > hi)


def fold_to_range(notes: list[QNote], lo: int, hi: int) -> tuple[list[QNote], FoldStats]:
    """把超出 [lo, hi] 的音折八度。先整句折（保留旋律輪廓），還是不行才逐音折。"""
    stats = FoldStats()
    out: list[QNote] = []
    for phrase in split_phrases(notes):
        stats.phrases += 1
        midis = [n.midi for n in phrase]
        best_shift, best_bad = 0, _out_of_range(midis, lo, hi)
        if best_bad:
            for shift in (-12, 12, -24, 24):
                bad = _out_of_range([m + shift for m in midis], lo, hi)
                if bad < best_bad or (bad == best_bad and abs(shift) < abs(best_shift) and best_shift != 0):
                    best_shift, best_bad = shift, bad
        if best_shift:
            stats.phrases_shifted += 1
        for n in phrase:
            m = n.midi + best_shift
            if m < lo or m > hi:
                stats.notes_shifted_individually += 1
                while m < lo:
                    m += 12
                while m > hi:
                    m -= 12
                if m < lo:  # 音域不足一個八度時才會發生
                    stats.notes_clipped += 1
                    m = lo if abs(m - lo) < abs(m + 12 - hi) else hi
            out.append(QNote(n.start, n.dur, m))
    return out, stats


# ---------------------------------------------------------------- 移調決策

def choose_transposition(written: list[QNote], lo: int, hi: int, key: KeyInfo, policy: str = "0") -> int:
    """policy:
      '0' / 整數 - 固定移這麼多半音（0 = 保留原調）。
      'auto'     - 在 -6..+6 半音中挑一個：先看折八度前超出音域的音最少，
                   再看調號升降記號最少，再看移動量最小。
    """
    if policy != "auto":
        return int(policy)
    midis = [n.midi for n in written]
    best = None
    for shift in range(-6, 7):
        bad = _out_of_range([m + shift for m in midis], lo, hi)
        accidentals = abs(key.transposed(shift).sharps)
        cand = (bad, accidentals, abs(shift), shift)
        if best is None or cand < best:
            best = cand
    return best[3]


def to_written(notes: list[QNote], transpose: int = 0) -> list[QNote]:
    return [QNote(n.start, n.dur, n.midi + CONCERT_TO_WRITTEN + transpose) for n in notes]
