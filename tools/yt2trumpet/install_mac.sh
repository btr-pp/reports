#!/bin/bash
# yt2trumpet 一鍵安裝（macOS）：建立虛擬環境、安裝套件與 ffmpeg。
# 用法：在終端機執行  bash install_mac.sh   （或在 Finder 對它按右鍵 → 打開方式 → 終端機）
set -e
cd "$(dirname "$0")"

echo "== 檢查 Python =="
PY=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    v=$("$c" -c 'import sys; print(sys.version_info[0]*100+sys.version_info[1])')
    if [ "$v" -ge 310 ] && [ "$v" -lt 313 ]; then PY="$c"; break; fi
  fi
done
if [ -z "$PY" ]; then
  echo "需要 Python 3.10～3.12。建議：brew install python@3.11"
  echo "（Basic Pitch 依賴的 TensorFlow 目前不支援 3.13）"
  exit 1
fi
echo "使用 $PY ($($PY --version))"

echo "== 建立虛擬環境 .venv =="
[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip >/dev/null

echo "== 安裝 yt2trumpet（含 Demucs、Basic Pitch、網頁介面；會下載約 2～3 GB，請耐心等）=="
pip install -e ".[all]"

echo "== ffmpeg =="
if command -v ffmpeg >/dev/null 2>&1; then
  echo "已有系統 ffmpeg：$(command -v ffmpeg)"
elif command -v brew >/dev/null 2>&1; then
  echo "用 Homebrew 安裝 ffmpeg…"; brew install ffmpeg || true
else
  echo "沒有 ffmpeg，會改用 imageio-ffmpeg 內附的版本（功能足夠）。"
fi

echo "== 先下載 Demucs 模型（約 80 MB），之後離線也能用 =="
python - <<'PY' || echo "模型下載失敗，第一次扒譜時會再試一次。"
from demucs.pretrained import get_model
get_model("htdemucs")
print("Demucs 模型就緒")
PY

echo
echo "安裝完成。之後雙擊 YT2Trumpet.command 就會開啟介面。"
