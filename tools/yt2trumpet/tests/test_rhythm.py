import numpy as np

from yt2trumpet import rhythm
from yt2trumpet.model import NoteEvent


def test_quantize_fixed_grid():
    beats = rhythm.fixed_grid(4.0, 120.0)
    notes = [NoteEvent(0.02, 0.48, 60), NoteEvent(0.51, 0.74, 62), NoteEvent(0.76, 1.0, 64), NoteEvent(1.0, 1.9, 65)]
    q, downbeat = rhythm.quantize(notes, beats, beats_per_bar=4, grid=4, downbeat=0)
    assert downbeat == 0
    assert [(n.start, n.dur, n.midi) for n in q] == [(0.0, 1.0, 60), (1.0, 0.5, 62), (1.5, 0.5, 64), (2.0, 1.75, 65)]


def test_quantize_pickup_goes_to_previous_bar():
    beats = rhythm.fixed_grid(4.0, 120.0)
    notes = [NoteEvent(0.0, 0.5, 60), NoteEvent(0.5, 1.5, 62)]
    q, _ = rhythm.quantize(notes, beats, beats_per_bar=4, grid=4, downbeat=1)
    # 拍點 #1 是強拍：第一個音在強拍前 → 放進弱起小節（start = 3）
    assert [(n.start, n.dur) for n in q] == [(3.0, 1.0), (4.0, 2.0)]


def test_quantize_removes_overlap_and_min_duration():
    beats = rhythm.fixed_grid(4.0, 120.0)
    notes = [NoteEvent(0.0, 1.2, 60), NoteEvent(1.0, 1.02, 62), NoteEvent(1.5, 2.0, 64)]
    q, _ = rhythm.quantize(notes, beats, grid=4, downbeat=0)
    assert [(n.start, n.dur, n.midi) for n in q] == [(0.0, 2.0, 60), (2.0, 0.25, 62), (3.0, 1.0, 64)]


def test_split_long_rests():
    from yt2trumpet.model import QNote

    q = [QNote(20.0, 1.0, 60), QNote(21.0, 1.0, 62), QNote(45.0, 1.0, 64)]
    out = rhythm.split_long_rests(q, max_rest_bars=2, beats_per_bar=4)
    assert out[0].start == 8.0            # 前奏 20 拍 → 壓成 2 小節
    assert out[2].start - out[1].end <= 8 + 4


def test_estimate_beats_on_synth(synth_audio):
    y, sr = synth_audio
    tempo, beats = rhythm.estimate_beats(y, sr)
    assert abs(tempo - 120) < 5 or abs(tempo - 60) < 3 or abs(tempo - 240) < 6
    assert len(beats) > 10


def test_quantize_drops_leading_empty_bars():
    beats = rhythm.fixed_grid(12.0, 120.0)
    notes = [NoteEvent(4.0, 4.5, 60), NoteEvent(4.5, 5.0, 62)]  # 第 8 拍才開始 = 前面 2 個空小節
    q, _ = rhythm.quantize(notes, beats, beats_per_bar=4, grid=4, downbeat=0)
    assert [n.start for n in q] == [0.0, 1.0]
