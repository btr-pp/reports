import pytest

from yt2trumpet import trumpet
from yt2trumpet.model import KeyInfo, QNote


def test_parse_note_and_range():
    assert trumpet.parse_note_name("C4") == 60
    assert trumpet.parse_note_name("Bb3") == 58
    assert trumpet.parse_note_name("F#5") == 78
    assert trumpet.parse_range("intermediate") == (55, 84)
    assert trumpet.parse_range("G3-C6") == (55, 84)
    with pytest.raises(ValueError):
        trumpet.parse_range("nonsense")


def test_to_written_adds_major_second():
    q = [QNote(0, 1, 60)]
    assert trumpet.to_written(q)[0].midi == 62
    assert trumpet.to_written(q, transpose=-2)[0].midi == 60


def test_key_info_names_and_sharps():
    assert KeyInfo(0, "major").sharps == 0
    assert KeyInfo(2, "major").name == "D major" and KeyInfo(2, "major").sharps == 2
    assert KeyInfo(10, "major").name == "Bb major" and KeyInfo(10, "major").sharps == -2
    assert KeyInfo(9, "minor").name == "a minor"
    assert KeyInfo(10, "major").transposed(2).name == "C major"


def test_fold_phrase_shift_keeps_contour():
    # 整句都太高：整句降八度，而不是逐音折
    q = [QNote(i, 1, m) for i, m in enumerate([86, 88, 90, 86])]
    out, stats = trumpet.fold_to_range(q, 55, 84)
    assert [n.midi for n in out] == [74, 76, 78, 74]
    assert stats.phrases_shifted == 1 and stats.notes_shifted_individually == 0


def test_fold_individual_note():
    q = [QNote(0, 1, 60), QNote(1, 1, 86), QNote(2, 1, 62)]
    out, stats = trumpet.fold_to_range(q, 55, 84)
    assert [n.midi for n in out] == [60, 74, 62]
    assert stats.phrases_shifted == 0 and stats.notes_shifted_individually == 1


def test_fold_splits_phrases_at_rests():
    q = [QNote(0, 1, 60), QNote(1, 1, 62), QNote(4, 1, 88), QNote(5, 1, 90)]
    out, stats = trumpet.fold_to_range(q, 55, 84)
    assert [n.midi for n in out] == [60, 62, 76, 78]
    assert stats.phrases == 2 and stats.phrases_shifted == 1


def test_choose_transposition_auto_prefers_in_range_then_few_accidentals():
    q = [QNote(i, 1, m) for i, m in enumerate([62, 64, 66, 69, 81])]
    key = KeyInfo(2, "major")  # D major（2 個升號）
    assert trumpet.choose_transposition(q, 55, 84, key, "0") == 0
    assert trumpet.choose_transposition(q, 55, 84, key, "-3") == -3
    shift = trumpet.choose_transposition(q, 60, 79, key, "auto")
    assert all(60 <= n.midi + shift <= 79 for n in q)
    assert key.transposed(shift).sharps == 0  # 移到 C 大調


def test_detect_key_from_notes_c_major():
    q = [QNote(i, 1, m) for i, m in enumerate([60, 62, 64, 65, 67, 69, 71, 72, 67, 64, 60])]
    k = trumpet.detect_key_from_notes(q)
    assert (k.tonic, k.mode) == (0, "major")
