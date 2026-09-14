"""網頁介面冒煙測試：先另開終端機執行 `yt2trumpet-ui --no-browser --port 7861`，再跑
    python bench/ui_smoke.py <音檔路徑>
會上傳音檔、選「不分離」、按開始，等結果出現後截圖 ui_after.png。需要 `pip install playwright` 與瀏覽器。"""
import glob
import os
import sys
import time

from playwright.sync_api import sync_playwright

wav = sys.argv[1]
port = int(os.environ.get("YT2TRUMPET_UI_PORT", "7861"))
exe = (glob.glob("/opt/pw-browsers/chromium-*/chrome-linux*/chrome") or [None])[0]
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
    pg = b.new_page(viewport={"width": 1400, "height": 1100})
    pg.goto(f"http://127.0.0.1:{port}", wait_until="networkidle")
    pg.set_input_files("input[type=file]", wav)
    time.sleep(2)
    pg.get_by_label("不分離（清唱 / 獨奏）").check()
    pg.get_by_role("button", name="開始扒譜").click()
    t0 = time.time()
    while time.time() - t0 < 600:
        txt = pg.inner_text("body")
        if "BPM" in txt or "失敗" in txt:
            break
        time.sleep(3)
    time.sleep(3)
    pg.screenshot(path="ui_after.png", full_page=True)
    body = pg.inner_text("body")
    ok = "BPM" in body and "失敗" not in body
    print("OK" if ok else "FAILED", f"({time.time() - t0:.0f}s)", "| 圖片數", pg.locator("img").count())
    b.close()
    sys.exit(0 if ok else 1)
