import numpy as np

from conftest import MELODY, expected_written
from yt2trumpet import pitch
from yt2trumpet.model import NoteEvent


def test_pyin_recovers_melody(synth_audio):
    y, sr = synth_audio
    notes = pitch.transcribe_pyin(y, sr)
    assert [n.midi for n in notes] == [m for m, _ in MELODY if m]


def test_f0_to_notes_hysteresis_absorbs_vibrato():
    times = np.arange(0, 1.0, 0.01)
    f0 = 440 * 2 ** (0.4 * np.sin(2 * np.pi * 6 * times) / 12)  # A4 ±0.4 半音抖音
    notes = pitch.f0_to_notes(times, f0, None)
    assert len(notes) == 1 and notes[0].midi == 69


def test_clean_notes_merges_and_drops():
    notes = [NoteEvent(0.0, 0.5, 60), NoteEvent(0.52, 1.0, 60), NoteEvent(1.0, 1.02, 62), NoteEvent(1.1, 1.5, 64)]
    out = pitch.clean_notes(notes, min_duration=0.07, max_gap=0.04)
    assert [(n.midi, round(n.onset, 2), round(n.offset, 2)) for n in out] == [(60, 0.0, 1.0), (64, 1.1, 1.5)]


def test_skyline_prefers_louder_then_higher():
    cands = [NoteEvent(0, 1, 60, 0.9), NoteEvent(0, 1, 72, 0.3), NoteEvent(1, 2, 64, 0.5), NoteEvent(1, 2, 67, 0.5)]
    out = pitch.skyline(cands)
    assert [n.midi for n in out] == [60, 67]
