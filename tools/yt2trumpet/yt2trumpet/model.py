"""共用資料結構。"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NoteEvent:
    """以秒為單位的音符事件（音高追蹤的輸出）。midi 為實音（concert pitch）。"""

    onset: float
    offset: float
    midi: int
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.offset - self.onset


@dataclass
class QNote:
    """量化後的音符：start / dur 皆以「拍」(四分音符) 為單位，自第一個強拍起算。"""

    start: float
    dur: float
    midi: int

    @property
    def end(self) -> float:
        return self.start + self.dur


@dataclass
class KeyInfo:
    """調性：tonic 為 0-11 的 pitch class（0 = C），mode 為 'major' / 'minor'。"""

    tonic: int
    mode: str

    NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

    def transposed(self, semitones: int) -> "KeyInfo":
        return KeyInfo((self.tonic + semitones) % 12, self.mode)

    @property
    def sharps(self) -> int:
        """調號的升記號數（負數 = 降記號）。以 -6..6 表示（Gb/F# 取 6）。"""
        major_tonic = self.tonic if self.mode == "major" else (self.tonic + 3) % 12
        # 五度圈：C=0, G=1, D=2 ... 以升號數表示
        circle = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: -5, 8: -4, 3: -3, 10: -2, 5: -1}
        return circle[major_tonic]

    @property
    def name(self) -> str:
        names = self.NAMES_FLAT if self.sharps < 0 else self.NAMES_SHARP
        tonic_name = names[self.tonic]
        if self.mode == "minor":
            return tonic_name.lower() + " minor"
        return tonic_name + " major"

    @property
    def tonic_name(self) -> str:
        names = self.NAMES_FLAT if self.sharps < 0 else self.NAMES_SHARP
        return names[self.tonic]


@dataclass
class TranscriptionResult:
    """整條 pipeline 的結果與統計。"""

    title: str
    out_dir: str
    tempo: float
    time_signature: str
    concert_key: KeyInfo
    written_key: KeyInfo
    transpose: int
    stem_used: str
    pitch_backend: str
    notes: list[QNote]
    files: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)
