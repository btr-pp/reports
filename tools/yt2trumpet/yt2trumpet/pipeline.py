"""把各步驟串起來。"""
from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf

from . import audio, backing, pitch, rhythm, score, separate, trumpet
from .model import KeyInfo, TranscriptionResult

log = logging.getLogger(__name__)


@dataclass
class Config:
    source: str                      # YouTube 網址或本機音檔
    out_dir: Path | None = None      # 預設 <out_root>/<標題>/
    out_root: Path | None = None     # 預設 ./output
    title: str | None = None
    start: float | None = None       # 裁切起點（秒）
    end: float | None = None
    stem: str = "auto"               # auto / vocals / other / none
    pitch_backend: str = "auto"      # auto / pyin / basic-pitch
    range_spec: str = "intermediate"  # beginner / intermediate / advanced / G3-C6
    transpose: str = "0"             # 半音數或 auto
    bpm: float | None = None
    beat_offset: float = 0.0         # 搭配 --bpm 使用：第一個強拍在第幾秒
    downbeat: int | None = None      # 自動偵測的拍點中，第幾個（0 起算）是第一個強拍；None = 自動猜
    time_signature: str = "4/4"
    grid: int = 4                    # 每拍等分數（4 = 十六分音符）
    min_note_ms: float = 70.0
    max_rest_bars: int = 2           # 超過這麼多小節的空白會被壓縮
    legato_beats: float = 1.0        # 短於此拍數的空隙補滿（換氣、音尾衰減不寫成休止符）
    formats: list[str] = field(default_factory=lambda: ["pdf", "png"])
    renderer: str = "auto"           # auto / musescore / verovio
    demucs_model: str = "htdemucs"
    device: str | None = None        # cpu / cuda / mps
    keep_temp: bool = False
    sr: int = 22050
    backing: bool = True             # 產生去掉主旋律的伴唱 / 伴奏音檔
    backing_format: str = "mp3"      # mp3 / wav
    progress: object = None          # callable(str)，給 UI 顯示進度用


def _report(cfg: "Config", msg: str) -> None:
    if cfg.progress:
        try:
            cfg.progress(msg)
        except Exception:
            pass


def run(cfg: Config) -> TranscriptionResult:
    warnings: list[str] = []
    tmp = Path(tempfile.mkdtemp(prefix="yt2trumpet_"))
    try:
        # 1. 取得音訊
        if audio.is_url(cfg.source):
            log.info(f"下載 {cfg.source} …")
            src_wav, yt_title = audio.download_audio(cfg.source, tmp)
            title = cfg.title or yt_title
        else:
            src = Path(cfg.source).expanduser()
            if not src.exists():
                raise FileNotFoundError(f"找不到檔案：{src}")
            src_wav = src
            title = cfg.title or src.stem
        safe_title = audio.safe_filename(title)
        out_dir = Path(cfg.out_dir) if cfg.out_dir else Path(cfg.out_root or "output") / safe_title
        out_dir.mkdir(parents=True, exist_ok=True)

        # 2. 裁切 + 轉成分析用 wav（44.1k 立體聲留給 demucs；分析用單聲道 22.05k）
        clip = tmp / "clip.wav"
        audio.to_wav(src_wav, clip, sr=44100, start=cfg.start, end=cfg.end)
        log.info(f"音訊長度 {sf.info(str(clip)).duration:.1f} 秒")

        # 3. 選軌（分離）
        _report(cfg, "分離人聲 / 伴奏…" if cfg.stem != "none" else "讀取音訊…")
        y_mel, stem_used, w, stems = separate.select_melody_audio(
            clip, tmp, mode=cfg.stem, sr=cfg.sr, model=cfg.demucs_model, device=cfg.device)
        warnings += w
        mel_wav = tmp / "melody_source.wav"
        sf.write(str(mel_wav), y_mel, cfg.sr)

        # 4. 音高追蹤
        log.info(f"追蹤音高（來源軌：{stem_used}）…")
        _report(cfg, f"追蹤音高（來源軌：{stem_used}）…")
        events, backend, w = pitch.transcribe(
            y_mel, cfg.sr, mel_wav, backend=cfg.pitch_backend, stem_used=stem_used,
            min_duration=cfg.min_note_ms / 1000.0)
        warnings += w
        if not events:
            raise RuntimeError("沒有偵測到任何音符；試試 --stem none / --pitch pyin，或縮小 --start/--end 範圍")
        log.info(f"偵測到 {len(events)} 個音符事件")

        # 5. 節拍與量化（節拍用原始混音抓比較穩）
        _report(cfg, "偵測節拍與量化…")
        y_mix, _ = audio.load_mono(clip, cfg.sr)
        beats_per_bar = int(cfg.time_signature.split("/")[0])
        if cfg.bpm:
            tempo = cfg.bpm
            beats = rhythm.fixed_grid(len(y_mix) / cfg.sr, cfg.bpm, cfg.beat_offset)
            downbeat = 0 if cfg.downbeat is None else cfg.downbeat  # offset 就是強拍
        else:
            env = rhythm.onset_envelope(y_mix, cfg.sr)
            tempo, beats = rhythm.track_beats(env, cfg.sr)
            beats, info = rhythm.refine_beats(beats, events, onset_env=env, sr=cfg.sr)
            if info:
                if abs(info["factor"] - 1.0) > 0.02 or info["phase"]:
                    log.info(f"依旋律起音校正拍點：{info['base_bpm']:.0f} → {info['bpm']:.0f} BPM（來源 {info['source']}），"
                             f"相位 {info['phase']:.3f} 拍，吻合度 {info['fit']:.2f}")
                tempo = info["bpm"]
            beats = rhythm.snap_grid_to_onsets(beats, events)
            downbeat = cfg.downbeat
        log.info(f"速度約 {tempo:.1f} BPM")
        audio_scores = rhythm.downbeat_scores_from_audio(y_mix, cfg.sr, beats, beats_per_bar) if downbeat is None else None
        qnotes, downbeat = rhythm.quantize(events, beats, beats_per_bar=beats_per_bar, grid=cfg.grid,
                                           downbeat=downbeat, audio_scores=audio_scores,
                                           legato_beats=cfg.legato_beats)
        log.info(f"第一個強拍：拍點 #{downbeat}（若小節線位置不對，用 --downbeat 0~{beats_per_bar - 1} 調整）")
        qnotes = rhythm.split_long_rests(qnotes, cfg.max_rest_bars, beats_per_bar)

        # 6. 調性
        try:
            concert_key = trumpet.detect_key_from_audio(y_mix, cfg.sr)
        except Exception:
            concert_key = trumpet.detect_key_from_notes(qnotes)

        # 7. 移調 → 記譜音 → 音域折疊
        lo, hi = trumpet.parse_range(cfg.range_spec)
        written0 = trumpet.to_written(qnotes, 0)
        written_key0 = concert_key.transposed(trumpet.CONCERT_TO_WRITTEN)
        shift = trumpet.choose_transposition(written0, lo, hi, written_key0, cfg.transpose)
        written = trumpet.to_written(qnotes, shift)
        written_key = written_key0.transposed(shift)
        written, fold = trumpet.fold_to_range(written, lo, hi)
        if shift:
            log.info(f"整首移調 {shift:+d} 半音（實音 {concert_key.name} → {concert_key.transposed(shift).name}）")
        if fold.phrases_shifted or fold.notes_shifted_individually:
            warnings.append(
                f"音域折疊：{fold.phrases_shifted}/{fold.phrases} 個樂句整句移八度，"
                f"{fold.notes_shifted_individually} 個音逐音移八度。")

        # 8. 樂譜
        _report(cfg, "排版樂譜…")
        sc = score.build_score(written, tempo, written_key, cfg.time_signature,
                               title=f"{title}（Bb 小號）", grid=cfg.grid)
        base = out_dir / safe_title
        files: dict[str, str] = {"musicxml": str(score.write_musicxml(sc, base.with_suffix(".musicxml")))}
        rendered, w, renderer = score.render(base.with_suffix(".musicxml"), base, cfg.formats, cfg.renderer)
        files.update(rendered)
        warnings += w

        # 9. 伴唱 / 伴奏
        if cfg.backing:
            _report(cfg, "產生伴奏音檔…")
            try:
                suffix = "伴唱" if stem_used == "vocals" else "伴奏"
                bpath, w = backing.make_backing(stems, stem_used, clip, out_dir / f"{safe_title}_{suffix}",
                                                transpose=shift, fmt=cfg.backing_format)
                files["backing"] = str(bpath)
                warnings += w
            except Exception as e:
                warnings.append(f"伴奏音檔產生失敗：{e}")
        if cfg.keep_temp:
            keep = out_dir / "work"
            shutil.copytree(tmp, keep, dirs_exist_ok=True)
            files["work_dir"] = str(keep)

        return TranscriptionResult(
            title=title, out_dir=str(out_dir), tempo=tempo, time_signature=cfg.time_signature,
            concert_key=concert_key, written_key=written_key, transpose=shift, stem_used=stem_used,
            pitch_backend=backend, notes=written, files=files, warnings=warnings,
            stats={"events": len(events), "notes": len(written), "renderer": renderer,
                   "lo": lo, "hi": hi},
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
