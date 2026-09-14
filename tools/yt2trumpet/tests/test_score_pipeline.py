from pathlib import Path

from conftest import expected_written
from yt2trumpet import score
from yt2trumpet.model import KeyInfo, QNote
from yt2trumpet.pipeline import Config, run


def test_build_score_and_musicxml(tmp_path):
    notes = [QNote(0, 1, 62), QNote(1, 0.5, 64), QNote(1.5, 0.5, 66), QNote(2, 2, 69), QNote(4, 4, 70)]
    sc = score.build_score(notes, 100, KeyInfo(10, "major"), "4/4", title="測試")
    path = score.write_musicxml(sc, tmp_path / "t.musicxml")
    assert path.exists()
    from music21 import converter

    p = converter.parse(str(path)).parts[0]
    got = [(n.pitch.midi, float(n.quarterLength)) for n in p.flatten().notes]
    assert got == [(62, 1.0), (64, 0.5), (66, 0.5), (69, 2.0), (70, 4.0)]
    assert p.flatten().notes[-1].pitch.name == "B-"  # 降記號調性用降記號拼法
    assert len(p.getElementsByClass("Measure")) == 2


def test_verovio_render_produces_files(tmp_path):
    notes = [QNote(i, 1, 60 + i) for i in range(8)]
    sc = score.build_score(notes, 120, KeyInfo(0, "major"), "4/4", title="測試（Bb 小號）")
    xml = score.write_musicxml(sc, tmp_path / "r.musicxml")
    files, warnings, renderer = score.render(xml, tmp_path / "r", ["pdf", "png"], renderer="verovio")
    assert renderer == "verovio"
    assert Path(files["pdf"]).stat().st_size > 1000
    assert Path(files["png"]).stat().st_size > 1000
    svg = Path(files["svg"]).read_text(encoding="utf-8")
    assert 'font-family="Leipzig"' not in svg and "" not in svg  # 字型已替換、SMuFL 節拍器音符已轉 Unicode


def test_end_to_end_on_synth(synth_wav, tmp_path):
    cfg = Config(source=str(synth_wav), out_dir=tmp_path / "out", stem="none", pitch_backend="pyin",
                 bpm=120.0, formats=["png"], renderer="verovio", title="e2e")
    res = run(cfg)
    assert [n.midi for n in res.notes] == expected_written()
    assert [n.start for n in res.notes][:5] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert res.written_key.name == "D major" and res.transpose == 0
    assert Path(res.files["musicxml"]).exists() and Path(res.files["png"]).exists()


def test_end_to_end_auto_beats_and_beginner(synth_wav, tmp_path):
    cfg = Config(source=str(synth_wav), out_dir=tmp_path / "out", stem="none", pitch_backend="pyin",
                 range_spec="beginner", transpose="auto", formats=[], title="e2e2")
    res = run(cfg)
    assert res.transpose == -2 and res.written_key.name == "C major"
    assert all(60 <= n.midi <= 79 for n in res.notes)
    assert len(res.notes) == len(expected_written())
