"""本機網頁介面（Gradio）。`yt2trumpet-ui` 啟動後會自動打開瀏覽器。"""
from __future__ import annotations

import logging
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from .pipeline import Config, run
from .trumpet import midi_to_name

log = logging.getLogger(__name__)

RANGE_CHOICES = {
    "初學（C4–G5）": "beginner",
    "中級（G3–C6）": "intermediate",
    "進階（F#3–E6）": "advanced",
}
STEM_CHOICES = {
    "自動判斷": "auto",
    "人聲是主旋律": "vocals",
    "樂器是主旋律": "other",
    "不分離（清唱 / 獨奏）": "none",
}
TRANSPOSE_CHOICES = {"保留原調": "0", "自動挑好吹的調": "auto"}


def _summary(res) -> str:
    flats = res.written_key.sharps < 0
    lo, hi = int(res.stats["lo"]), int(res.stats["hi"])
    lines = [
        f"**{res.title}**",
        f"- 速度 {res.tempo:.0f} BPM，拍號 {res.time_signature}",
        f"- 實音 {res.concert_key.name} → 小號記譜 {res.written_key.name}"
        + (f"（整首移調 {res.transpose:+d} 半音）" if res.transpose else "（保留原調）"),
        f"- 旋律來源：{res.stem_used} 軌，音高後端：{res.pitch_backend}，音符 {res.stats['notes']} 個，"
        f"音域限制 {midi_to_name(lo, flats)}–{midi_to_name(hi, flats)}",
        f"- 輸出資料夾：`{res.out_dir}`",
    ]
    if res.warnings:
        lines.append("\n**提醒**")
        lines += [f"- {w}" for w in res.warnings]
    return "\n".join(lines)


def transcribe(url, upload, start, end, stem, rng, transpose, semitones, bpm, downbeat, time_sig, grid,
               legato, title, backing, out_root, progress=None):
    import gradio as gr

    progress = progress or gr.Progress()
    source = (url or "").strip() or (upload if upload else "")
    if not source:
        raise gr.Error("請貼 YouTube 網址或上傳音檔")
    tr = TRANSPOSE_CHOICES.get(transpose, "0")
    if tr == "0" and semitones:
        tr = str(int(semitones))

    steps = {"分離": 0.15, "追蹤": 0.45, "節拍": 0.7, "排版": 0.85, "伴奏": 0.92}

    def report(msg):
        frac = next((v for k, v in steps.items() if k in msg), 0.3)
        progress(frac, desc=msg)

    progress(0.02, desc="開始…")
    cfg = Config(
        source=source, out_root=Path(out_root) if out_root else None, title=title or None,
        start=start or None, end=end or None, stem=STEM_CHOICES.get(stem, "auto"),
        range_spec=RANGE_CHOICES.get(rng, "intermediate"), transpose=tr,
        bpm=bpm or None, downbeat=int(downbeat) if downbeat not in (None, "", "自動") else None,
        time_signature=time_sig or "4/4", grid=int(grid), legato_beats=float(legato),
        formats=["pdf", "png"], backing=bool(backing), progress=report,
    )
    try:
        res = run(cfg)
    except Exception as e:
        log.error(traceback.format_exc())
        raise gr.Error(f"失敗：{e}")
    progress(1.0, desc="完成")

    pngs = [v for k, v in sorted(res.files.items()) if k.startswith("png")]
    pdf = res.files.get("pdf")
    xml = res.files.get("musicxml")
    back = res.files.get("backing")
    downloads = [f for f in (pdf, xml, back) if f]
    return _summary(res), pngs, back, downloads


def build_app():
    import gradio as gr

    with gr.Blocks(title="yt2trumpet 小號扒譜") as app:
        gr.Markdown("# 🎺 yt2trumpet\n把 YouTube 或音檔的主旋律扒成 **Bb 小號五線譜**，並產生去掉主旋律的伴奏。")
        with gr.Row():
            with gr.Column(scale=1):
                url = gr.Textbox(label="YouTube 網址", placeholder="https://www.youtube.com/watch?v=…")
                upload = gr.Audio(label="或上傳音檔", type="filepath", sources=["upload"])
                with gr.Row():
                    start = gr.Number(label="起點（秒）", value=None, precision=1)
                    end = gr.Number(label="終點（秒）", value=None, precision=1)
                stem = gr.Radio(list(STEM_CHOICES), value="自動判斷", label="主旋律來源")
                rng = gr.Radio(list(RANGE_CHOICES), value="中級（G3–C6）", label="吹奏程度 / 記譜音域")
                with gr.Row():
                    transpose = gr.Radio(list(TRANSPOSE_CHOICES), value="保留原調", label="調性",
                                         info="也可以在右邊直接填半音數（例如 -2）")
                    semitones = gr.Number(label="手動移調（半音）", value=0, precision=0)
                backing = gr.Checkbox(label="一起產生伴奏音檔（去掉主旋律）", value=True)
                with gr.Accordion("進階選項", open=False):
                    with gr.Row():
                        bpm = gr.Number(label="固定 BPM（空白 = 自動）", value=None, precision=1)
                        downbeat = gr.Dropdown(["自動", "0", "1", "2", "3"], value="自動", label="第一個強拍在第幾個拍點")
                    with gr.Row():
                        time_sig = gr.Dropdown(["4/4", "3/4", "2/4", "6/8"], value="4/4", label="拍號")
                        grid = gr.Dropdown(["4", "2", "3"], value="4", label="最小音符（4=十六分, 2=八分, 3=三連音）")
                    legato = gr.Slider(0, 2, value=1.0, step=0.25, label="補滿短於幾拍的空隙（換氣不寫休止符）")
                    title = gr.Textbox(label="樂譜標題（空白 = 影片標題）")
                    out_root = gr.Textbox(label="輸出資料夾", value=str(Path.cwd() / "output"))
                btn = gr.Button("開始扒譜", variant="primary")
            with gr.Column(scale=2):
                summary = gr.Markdown("結果會顯示在這裡。")
                gallery = gr.Gallery(label="樂譜預覽", columns=1, height=700, object_fit="contain")
                audio = gr.Audio(label="伴奏試聽", type="filepath")
                files = gr.File(label="下載（PDF / MusicXML / 伴奏）", file_count="multiple")
        btn.click(transcribe,
                  inputs=[url, upload, start, end, stem, rng, transpose, semitones, bpm, downbeat, time_sig, grid,
                          legato, title, backing, out_root],
                  outputs=[summary, gallery, audio, files])
    return app


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="yt2trumpet-ui", description="啟動 yt2trumpet 網頁介面")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--no-browser", action="store_true", help="不要自動打開瀏覽器")
    p.add_argument("--share", action="store_true", help="產生 Gradio 公開連結（給別台裝置用）")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("numba", "urllib3", "httpx", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    app = build_app()
    app.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=args.port, inbrowser=not args.no_browser, share=args.share,
        show_error=True, quiet=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
