"""樂譜輸出：QNote → music21 → MusicXML → PDF / PNG。

渲染優先順序：
  1. MuseScore（若系統有安裝，品質最好）
  2. verovio（純 pip）產 SVG，再以 cairosvg 轉 PDF / PNG
  3. 只留 MusicXML 與 SVG
"""
from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from .model import KeyInfo, QNote

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- music21

def _spell(midi: int, key_sharps: int):
    from music21 import pitch

    p = pitch.Pitch(midi=midi)
    if key_sharps < 0 and p.accidental is not None and p.accidental.name == "sharp":
        p = p.getEnharmonic()
    return p


def build_score(
    notes: list[QNote],
    tempo: float,
    written_key: KeyInfo,
    time_signature: str = "4/4",
    title: str = "",
    composer: str = "yt2trumpet",
    grid: int = 4,
):
    from music21 import clef, instrument, key, metadata, meter, note, stream, tempo as m21tempo

    sc = stream.Score()
    sc.metadata = metadata.Metadata()
    sc.metadata.title = title or "Untitled"
    sc.metadata.composer = composer

    part = stream.Part()
    part.id = "Trumpet"
    inst = instrument.Trumpet()
    inst.partName = "Bb Trumpet"
    part.insert(0, inst)
    part.insert(0, clef.TrebleClef())
    part.insert(0, meter.TimeSignature(time_signature))
    part.insert(0, key.KeySignature(written_key.sharps))
    part.insert(0, m21tempo.MetronomeMark(number=int(round(tempo))))

    for q in notes:
        n = note.Note()
        n.pitch = _spell(q.midi, written_key.sharps)
        n.duration.quarterLength = Fraction(q.dur).limit_denominator(grid * 3)
        part.insert(Fraction(q.start).limit_denominator(grid * 3), n)

    part.makeMeasures(inPlace=True)
    part.makeRests(fillGaps=True, inPlace=True, timeRangeFromBarDuration=True)
    part.makeTies(inPlace=True)
    part.makeBeams(inPlace=True)
    part.makeAccidentals(inPlace=True)
    sc.insert(0, part)
    return sc


def write_musicxml(score, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=str(path))
    return path


# ---------------------------------------------------------------- 渲染

MUSESCORE_CANDIDATES = [
    "mscore", "musescore", "mscore4", "musescore4", "MuseScore4", "mscore4portable", "mscore3", "musescore3",
]
MUSESCORE_PATHS = {
    "Darwin": [
        "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
        "/Applications/MuseScore 3.app/Contents/MacOS/mscore",
    ],
    "Windows": [
        r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
        r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe",
    ],
    "Linux": ["/usr/bin/mscore", "/usr/bin/musescore", "/snap/bin/musescore"],
}


def find_musescore() -> str | None:
    env = os.environ.get("MUSESCORE")
    if env and Path(env).exists():
        return env
    for c in MUSESCORE_CANDIDATES:
        p = shutil.which(c)
        if p:
            return p
    for p in MUSESCORE_PATHS.get(platform.system(), []):
        if Path(p).exists():
            return p
    return None


def render_with_musescore(mscore: str, musicxml: Path, out_base: Path, formats: list[str]) -> dict[str, str]:
    files: dict[str, str] = {}
    for fmt in formats:
        target = out_base.with_suffix(f".{fmt}")
        subprocess.run([mscore, "-o", str(target), str(musicxml)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if fmt == "png":
            pages = sorted(out_base.parent.glob(f"{out_base.name}-*.png"))
            if pages:
                files["png"] = str(pages[0])
                for i, p in enumerate(pages[1:], start=2):
                    files[f"png_page{i}"] = str(p)
        elif target.exists():
            files[fmt] = str(target)
    return files


def render_svgs_verovio(musicxml: Path, scale: int = 40) -> list[str]:
    import verovio

    tk = verovio.toolkit()
    tk.setOptions({
        "pageWidth": 2100, "pageHeight": 2970,  # A4 直式（單位為 MEI 的 1/10 mm）
        "scale": scale,
        "adjustPageHeight": False,
        "breaks": "auto",
        "header": "auto", "footer": "none",
        "spacingStaff": 8, "spacingSystem": 10,
        "font": "Leipzig",
    })
    if not tk.loadFile(str(musicxml)):
        raise RuntimeError("verovio 無法讀取 MusicXML")
    return [tk.renderToSVG(p) for p in range(1, tk.getPageCount() + 1)]


# cairosvg 只會用 font-family 清單裡的第一個字型，且不支援 SVG 內嵌字型，
# 所以把所有文字改用一個「同時有中文與 ♩ 符號」的系統字型，並把 SMuFL 私有區的節拍器音符換成 Unicode。
CJK_FONT_CANDIDATES = {
    "Darwin": ["PingFang TC", "PingFang SC", "Heiti TC", "Arial Unicode MS"],
    "Windows": ["Microsoft JhengHei", "Microsoft YaHei", "SimSun", "Arial Unicode MS"],
    "Linux": ["Noto Sans CJK TC", "Noto Sans CJK SC", "Noto Sans TC", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
              "Source Han Sans TC", "Droid Sans Fallback"],
}
_SMUFL_TO_UNICODE = {
    "\uE1D5": "\u2669",  # 四分音符 → ♩
    "\uE1D7": "\u266A",  # 八分音符 → ♪
    "\uECA5": "\u2669",  # Leipzig 節拍器四分音符
    "\uECA7": "\u266A",  # Leipzig 節拍器八分音符
    "\uECA3": "\U0001D15E",  # 二分音符
    "\uECB7": ".",        # 附點
}


def pick_text_font() -> str:
    env = os.environ.get("YT2TRUMPET_FONT")
    if env:
        return env
    cands = CJK_FONT_CANDIDATES.get(platform.system(), CJK_FONT_CANDIDATES["Linux"])
    fc = shutil.which("fc-list")
    if fc:
        try:
            installed = subprocess.run([fc, ":", "family"], capture_output=True, text=True, timeout=20).stdout
            for c in cands:
                if c in installed:
                    return c
        except Exception:
            pass
    return cands[0]


def patch_svg_fonts(svg: str, font: str) -> str:
    import re

    for k, v in _SMUFL_TO_UNICODE.items():
        svg = svg.replace(k, v)
    svg = re.sub(r'font-family="(Times, serif|Leipzig|serif|sans-serif)"', f'font-family="{font}"', svg)
    return svg


def render_with_verovio(musicxml: Path, out_base: Path, formats: list[str]) -> tuple[dict[str, str], list[str]]:
    files: dict[str, str] = {}
    warnings: list[str] = []
    font = pick_text_font()
    svgs = [patch_svg_fonts(s, font) for s in render_svgs_verovio(musicxml)]
    svg_paths = []
    for i, svg in enumerate(svgs, start=1):
        p = out_base.parent / f"{out_base.name}-{i}.svg"
        p.write_text(svg, encoding="utf-8")
        svg_paths.append(p)
    files["svg"] = str(svg_paths[0])

    try:
        import cairosvg
    except Exception:
        warnings.append("未安裝 cairosvg（或系統缺 cairo），只輸出 SVG 與 MusicXML；`pip install cairosvg` 後可產 PDF/PNG。")
        return files, warnings

    if "pdf" in formats:
        pdf = out_base.with_suffix(".pdf")
        if len(svgs) == 1:
            cairosvg.svg2pdf(bytestring=svgs[0].encode("utf-8"), write_to=str(pdf))
        else:
            _merge_svgs_to_pdf(svgs, pdf)
        files["pdf"] = str(pdf)
    if "png" in formats:
        for i, svg in enumerate(svgs, start=1):
            png = out_base.parent / f"{out_base.name}-{i}.png"
            cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(png), output_width=1654)
            files["png" if i == 1 else f"png_page{i}"] = str(png)
    return files, warnings


def _merge_svgs_to_pdf(svgs: list[str], pdf: Path) -> None:
    """多頁：每頁各轉一個 PDF 再合併（pypdf 有裝就合併，沒有就只留第一頁並警告）。"""
    import io

    import cairosvg

    pages = [cairosvg.svg2pdf(bytestring=s.encode("utf-8")) for s in svgs]
    try:
        from pypdf import PdfReader, PdfWriter
    except Exception:
        try:
            from PyPDF2 import PdfReader, PdfWriter  # type: ignore
        except Exception:
            pdf.write_bytes(pages[0])
            log.warning("未安裝 pypdf，PDF 只含第一頁（其餘頁請看 PNG/SVG）")
            return
    writer = PdfWriter()
    for data in pages:
        for page in PdfReader(io.BytesIO(data)).pages:
            writer.add_page(page)
    with open(pdf, "wb") as f:
        writer.write(f)


def render(musicxml: Path, out_base: Path, formats: list[str], renderer: str = "auto") -> tuple[dict[str, str], list[str], str]:
    """回傳 (檔案表, 警告, 實際使用的渲染器)。"""
    wanted = [f for f in formats if f in ("pdf", "png")]
    if not wanted:
        return {}, [], "none"
    if renderer in ("auto", "musescore"):
        ms = find_musescore()
        if ms:
            try:
                return render_with_musescore(ms, musicxml, out_base, wanted), [], "musescore"
            except Exception as e:
                if renderer == "musescore":
                    raise
                log.warning(f"MuseScore 渲染失敗（{e}），改用 verovio")
        elif renderer == "musescore":
            raise RuntimeError("找不到 MuseScore，請安裝或設定環境變數 MUSESCORE 指向執行檔")
    files, warnings = render_with_verovio(musicxml, out_base, wanted)
    return files, warnings, "verovio"
