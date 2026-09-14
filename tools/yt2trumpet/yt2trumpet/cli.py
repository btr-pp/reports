"""命令列介面。"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .pipeline import Config, run
from .trumpet import midi_to_name


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="yt2trumpet",
        description="把 YouTube 影片或音檔的主旋律扒成 Bb 小號五線譜（PDF/PNG/MusicXML）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""範例：
  yt2trumpet "https://www.youtube.com/watch?v=xxxx"
  yt2trumpet song.mp3 --start 60 --end 95 --stem vocals
  yt2trumpet song.wav --bpm 120 --range beginner --transpose auto
""",
    )
    p.add_argument("source", help="YouTube 網址或本機音檔（mp3/m4a/wav…）")
    p.add_argument("-o", "--out", type=Path, help="輸出資料夾（預設 ./output/<標題>/）")
    p.add_argument("--title", help="樂譜標題（預設用影片標題或檔名）")
    p.add_argument("--start", type=float, help="裁切起點（秒）")
    p.add_argument("--end", type=float, help="裁切終點（秒）")
    p.add_argument("--stem", choices=["auto", "vocals", "other", "none"], default="auto",
                   help="旋律來源軌：auto 自動判斷 / vocals 人聲 / other 旋律樂器 / none 不分離（預設 auto）")
    p.add_argument("--pitch", choices=["auto", "pyin", "basic-pitch"], default="auto",
                   help="音高追蹤後端（預設 auto：人聲用 pyin，其他有 basic-pitch 就用）")
    p.add_argument("--range", default="intermediate",
                   help="記譜音域：beginner(C4-G5) / intermediate(G3-C6) / advanced(F#3-E6) 或自訂如 G3-C6")
    p.add_argument("--transpose", default="0",
                   help="整首移調半音數（0 = 保留原調，auto = 自動挑好吹的調）")
    p.add_argument("--bpm", type=float, help="手動指定速度；不指定則自動偵測")
    p.add_argument("--offset", type=float, default=0.0, help="搭配 --bpm：第一個強拍落在第幾秒")
    p.add_argument("--downbeat", type=int, help="自動偵測拍點時，第幾個拍點是第一個強拍（0~拍數-1）；不給就自動猜")
    p.add_argument("--time", default="4/4", help="拍號（預設 4/4）")
    p.add_argument("--grid", type=int, default=4, help="每拍等分數：4 = 十六分音符，2 = 八分音符，3 = 三連音")
    p.add_argument("--min-note", type=float, default=70.0, help="最短音符（毫秒），更短的視為雜訊")
    p.add_argument("--max-rest-bars", type=int, default=2, help="超過此小節數的空白會被壓縮（0 = 不壓縮）")
    p.add_argument("--legato", type=float, default=1.0,
                   help="短於此拍數的空隙視為換氣，前一個音延長補滿；0 = 忠實保留所有休止符（預設 1）")
    p.add_argument("--formats", default="pdf,png", help="輸出格式，逗號分隔：pdf,png（MusicXML 一定會輸出）")
    p.add_argument("--renderer", choices=["auto", "musescore", "verovio"], default="auto")
    p.add_argument("--demucs-model", default="htdemucs")
    p.add_argument("--device", help="demucs 裝置：cpu / cuda / mps")
    p.add_argument("--no-backing", action="store_true", help="不要產生去掉主旋律的伴唱 / 伴奏音檔")
    p.add_argument("--backing-format", choices=["mp3", "wav"], default="mp3")
    p.add_argument("--keep-temp", action="store_true", help="保留下載與分離的中間檔到輸出資料夾 work/")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(message)s" if not args.verbose else "%(levelname)s %(name)s: %(message)s")
    for noisy in ("numba", "urllib3", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = Config(
        source=args.source, out_dir=args.out, title=args.title, start=args.start, end=args.end,
        stem=args.stem, pitch_backend=args.pitch, range_spec=args.range, transpose=args.transpose,
        bpm=args.bpm, beat_offset=args.offset, downbeat=args.downbeat, time_signature=args.time, grid=args.grid,
        min_note_ms=args.min_note, max_rest_bars=args.max_rest_bars, legato_beats=args.legato,
        formats=[f.strip() for f in args.formats.split(",") if f.strip()],
        renderer=args.renderer, demucs_model=args.demucs_model, device=args.device, keep_temp=args.keep_temp,
        backing=not args.no_backing, backing_format=args.backing_format,
    )
    try:
        res = run(cfg)
    except Exception as e:
        if args.verbose:
            raise
        print(f"錯誤：{e}", file=sys.stderr)
        return 1

    flats = res.written_key.sharps < 0
    lo, hi = int(res.stats["lo"]), int(res.stats["hi"])
    print()
    print(f"標題：{res.title}")
    print(f"速度：{res.tempo:.0f} BPM，拍號 {res.time_signature}")
    print(f"實音調性：{res.concert_key.name} → 小號記譜調性：{res.written_key.name}"
          + (f"（整首移調 {res.transpose:+d} 半音）" if res.transpose else "（保留原調）"))
    print(f"旋律來源：{res.stem_used} 軌，音高後端：{res.pitch_backend}，渲染：{res.stats['renderer']}")
    print(f"音符數：{res.stats['notes']}，記譜音域限制 {midi_to_name(lo, flats)}–{midi_to_name(hi, flats)}")
    print("輸出檔案：")
    for k, v in res.files.items():
        print(f"  {k:10s} {v}")
    if res.warnings:
        print("提醒：")
        for w in res.warnings:
            print(f"  - {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
