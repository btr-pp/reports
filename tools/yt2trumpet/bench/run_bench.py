"""跑基準測試：合成 → pipeline → 與標準答案比對。

用法：python bench/run_bench.py [--scenarios clean,vocal,...] [--pieces N] [--known-bpm] [--pitch pyin]
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from synth import SCENARIOS, SR, score_to_events, synth  # noqa: E402

from yt2trumpet.pipeline import Config, run  # noqa: E402
from yt2trumpet.trumpet import CONCERT_TO_WRITTEN  # noqa: E402


def load_pieces(n: int):
    """從 music21 語料庫挑公有領域旋律：Essen 民歌 + 巴哈聖詠女高音。"""
    from music21 import corpus

    pieces = []
    for path in corpus.getPaths(fileExtensions=["abc"]):
        if "essenFolksong" not in str(path):
            continue
        opus = corpus.parse(path)
        for sc in getattr(opus, "scores", [opus]):
            try:
                events, bpb, key = score_to_events(sc)
            except Exception:
                continue
            if bpb not in (3, 4) or not (16 <= len(events) <= 60):
                continue
            durs = [d for _, d, _ in events]
            if min(durs) < 0.25:
                continue
            title = (sc.metadata.title if sc.metadata else None) or path.stem
            pieces.append((f"essen/{title}", events, bpb, key))
            if len(pieces) >= n:
                return pieces
        if len(pieces) >= n:
            break
    return pieces


def evaluate(truth, got, bpb=4, tol=0.2):
    """truth/got: [(start_beat, dur, midi)]。回傳 pitch 序列相似度、onset+pitch F1、拍位偏移估計。"""
    t_m = [m for _, _, m in truth]
    g_m = [m for _, _, m in got]
    seq = difflib.SequenceMatcher(a=t_m, b=g_m).ratio()
    # 最佳整體拍位偏移（以 0.25 拍為單位，-8..8 拍）：模擬小節線 / 弱起判斷錯誤
    best = (0, -1)
    for shift in np.arange(-16, 16.01, 0.25):
        used = set()
        tp = 0
        for s, d, m in truth:
            for j, (gs, gd, gm) in enumerate(got):
                if j in used:
                    continue
                if gm == m and abs(gs - shift - s) <= tol:
                    used.add(j)
                    tp += 1
                    break
        if tp > best[1] or (tp == best[1] and abs(shift) < abs(best[0])):
            best = (float(shift), tp)
    shift, tp = best
    prec = tp / max(len(got), 1)
    rec = tp / max(len(truth), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"seq": seq, "f1": f1, "prec": prec, "rec": rec, "shift": shift, "n_truth": len(truth), "n_got": len(got),
            "bar_ok": float(shift % bpb == 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--pieces", type=int, default=6)
    ap.add_argument("--known-bpm", action="store_true")
    ap.add_argument("--pitch", default="pyin")
    ap.add_argument("--out", type=Path, default=Path("bench/out"))
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    pieces = load_pieces(args.pieces)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sc_name in args.scenarios.split(","):
        sc = SCENARIOS[sc_name]
        for name, events, bpb, key in pieces:
            y, truth, _ = synth(events, bpb, key, sc)
            safe = name.replace("/", "_").replace(" ", "_")[:40]
            wav = args.out / f"{sc_name}__{safe}.wav"
            sf.write(str(wav), y, SR)
            cfg = Config(source=str(wav), out_dir=args.out / f"{sc_name}__{safe}", stem="none",
                         pitch_backend=args.pitch, formats=[], time_signature=f"{bpb}/4",
                         bpm=sc.bpm if args.known_bpm else None, max_rest_bars=0, title=safe)
            t0 = time.time()
            try:
                res = run(cfg)
                got = [(n.start, n.dur, n.midi - CONCERT_TO_WRITTEN - res.transpose) for n in res.notes]
                ev = evaluate(truth, got, bpb)
                ev["tempo"] = res.tempo
            except Exception as e:
                ev = {"seq": 0, "f1": 0, "prec": 0, "rec": 0, "shift": 0, "n_truth": len(truth), "n_got": 0, "bar_ok": 0.0, "err": str(e)[:60]}
            ev.update(scenario=sc_name, piece=name, bpb=bpb, secs=time.time() - t0)
            rows.append(ev)
            print(f"{sc_name:12s} {name[:34]:34s} {bpb}/4 seq={ev['seq']:.2f} f1={ev['f1']:.2f} "
                  f"shift={ev['shift']:+.2f} n={ev['n_got']}/{ev['n_truth']} tempo={ev.get('tempo', 0):.0f} "
                  f"{ev.get('err', '')}", flush=True)

    print("\n=== 各情境平均 ===")
    for sc_name in args.scenarios.split(","):
        rs = [r for r in rows if r["scenario"] == sc_name]
        print(f"{sc_name:12s} seq={np.mean([r['seq'] for r in rs]):.3f} f1={np.mean([r['f1'] for r in rs]):.3f} "
              f"bar_ok={np.mean([r['bar_ok'] for r in rs]):.2f} secs={np.mean([r['secs'] for r in rs]):.1f}")
    if args.json:
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
