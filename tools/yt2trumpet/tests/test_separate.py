import numpy as np
import soundfile as sf

from yt2trumpet import separate


def _fake_stems(tmp_path, vocal_gain, other_gain, sr=22050):
    rng = np.random.default_rng(1)
    paths = {}
    for name, gain in (("vocals", vocal_gain), ("drums", 0.15), ("bass", 0.15), ("other", other_gain)):
        y = (gain * rng.standard_normal(sr)).astype(np.float32)
        p = tmp_path / f"{name}.wav"
        sf.write(str(p), y, sr)
        paths[name] = p
    return paths


def test_vocal_ratio(tmp_path):
    stems = _fake_stems(tmp_path, vocal_gain=0.5, other_gain=0.15)
    assert separate.vocal_ratio(stems) > 0.5
    stems = _fake_stems(tmp_path, vocal_gain=0.01, other_gain=0.5)
    assert separate.vocal_ratio(stems) < 0.05


def test_auto_picks_vocals_when_singing(tmp_path, monkeypatch, synth_wav):
    stems = _fake_stems(tmp_path, vocal_gain=0.5, other_gain=0.15)
    monkeypatch.setattr(separate, "demucs_available", lambda: True)
    monkeypatch.setattr(separate, "run_demucs", lambda *a, **k: stems)
    y, used, warnings, got = separate.select_melody_audio(synth_wav, tmp_path, mode="auto")
    assert used == "vocals" and warnings == [] and len(y) == 22050 and got is stems


def test_auto_picks_other_when_instrumental(tmp_path, monkeypatch, synth_wav):
    stems = _fake_stems(tmp_path, vocal_gain=0.01, other_gain=0.5)
    monkeypatch.setattr(separate, "demucs_available", lambda: True)
    monkeypatch.setattr(separate, "run_demucs", lambda *a, **k: stems)
    y, used, _, got = separate.select_melody_audio(synth_wav, tmp_path, mode="auto")
    assert used == "other" and got is stems


def test_none_and_missing_demucs_fall_back_to_mix(tmp_path, monkeypatch, synth_wav):
    y, used, warnings, got = separate.select_melody_audio(synth_wav, tmp_path, mode="none")
    assert used == "mix" and warnings == [] and got is None
    monkeypatch.setattr(separate, "demucs_available", lambda: False)
    y, used, warnings, got = separate.select_melody_audio(synth_wav, tmp_path, mode="auto")
    assert used == "mix" and len(warnings) == 1 and got is None
