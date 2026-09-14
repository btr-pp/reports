import numpy as np
import pytest
import soundfile as sf

SR = 22050
BPM = 120.0

# 原創測試旋律（實音 MIDI, 拍數），規整 4/4
MELODY = [(60, 1), (62, 1), (64, 1), (67, 1),
          (69, 0.5), (67, 0.5), (64, 1), (62, 2),
          (60, 2), (0, 2),
          (67, 1), (72, 1), (74, 1), (76, 0.5), (74, 0.5),
          (72, 1), (81, 1), (79, 1), (77, 1),
          (76, 2), (74, 2),
          (72, 4)]


def synth_melody(melody=MELODY, bpm=BPM, sr=SR, click=True):
    beat = 60 / bpm

    def tone(m, dur):
        n = int(dur * sr)
        t = np.arange(n) / sr
        if m == 0:
            return np.zeros(n)
        f = 440 * 2 ** ((m - 69) / 12)
        env = np.minimum(1, t / 0.01) * np.minimum(1, (dur - t) / 0.05)
        return 0.4 * env * (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t))

    y = np.concatenate([tone(m, d * beat) for m, d in melody])
    if click:
        rng = np.random.default_rng(0)
        for b in range(int(sum(d for _, d in melody))):
            i = int(b * beat * sr)
            y[i:i + 200] += 0.2 * rng.standard_normal(200) * np.exp(-np.arange(200) / 40)
    return y.astype(np.float32), sr


@pytest.fixture(scope="session")
def synth_audio():
    return synth_melody()


@pytest.fixture(scope="session")
def synth_wav(tmp_path_factory, synth_audio):
    y, sr = synth_audio
    p = tmp_path_factory.mktemp("audio") / "synth.wav"
    sf.write(str(p), y, sr)
    return p


def expected_written():
    return [m + 2 for m, _ in MELODY if m]
