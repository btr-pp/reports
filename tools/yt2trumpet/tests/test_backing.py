import numpy as np
import soundfile as sf

from yt2trumpet import backing


def _stems(tmp_path, sr=22050):
    t = np.arange(sr) / sr
    paths = {}
    for name, f, gain in (("vocals", 440, 0.5), ("drums", 0, 0.0), ("bass", 110, 0.3), ("other", 330, 0.2)):
        y = gain * np.sin(2 * np.pi * f * t) if f else np.zeros(sr)
        p = tmp_path / f"{name}.wav"
        sf.write(str(p), np.stack([y, y], axis=1).astype(np.float32), sr)
        paths[name] = p
    return paths


def test_backing_removes_melody_stem(tmp_path):
    stems = _stems(tmp_path)
    y, sr = backing.backing_from_stems(stems, "vocals")
    spec = np.abs(np.fft.rfft(y[:, 0]))
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    assert spec[np.argmin(abs(freqs - 440))] < 0.01 * spec.max()   # 人聲 440Hz 不見了
    assert spec[np.argmin(abs(freqs - 110))] > 0.5 * spec.max()    # 貝斯還在
    y2, _ = backing.backing_from_stems(stems, "other")
    spec2 = np.abs(np.fft.rfft(y2[:, 0]))
    assert spec2[np.argmin(abs(freqs - 330))] < 0.01 * spec2.max()


def test_center_cancel_removes_center(tmp_path):
    sr = 22050
    t = np.arange(sr) / sr
    center = 0.5 * np.sin(2 * np.pi * 440 * t)
    side = 0.3 * np.sin(2 * np.pi * 220 * t)
    p = tmp_path / "stereo.wav"
    sf.write(str(p), np.stack([center + side, center - side], axis=1).astype(np.float32), sr)
    y, _ = backing.center_cancel(p)
    spec = np.abs(np.fft.rfft(y[:, 0]))
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    assert spec[np.argmin(abs(freqs - 440))] < 0.01 * spec[np.argmin(abs(freqs - 220))]


def test_make_backing_transposes_and_exports_wav(tmp_path):
    stems = _stems(tmp_path)
    out, warnings = backing.make_backing(stems, "vocals", tmp_path / "x.wav", tmp_path / "b", transpose=2, fmt="wav")
    assert out.exists() and out.suffix == ".wav" and warnings == []
    y, sr = sf.read(str(out))
    spec = np.abs(np.fft.rfft(y[:, 0]))
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    peak = freqs[np.argmax(spec)]
    assert abs(peak - 110 * 2 ** (2 / 12)) < 3  # 貝斯 110Hz 升了大二度


def test_make_backing_mp3(tmp_path):
    stems = _stems(tmp_path)
    out, _ = backing.make_backing(stems, "vocals", tmp_path / "x.wav", tmp_path / "b", fmt="mp3")
    assert out.suffix in (".mp3", ".wav") and out.stat().st_size > 1000
