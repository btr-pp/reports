#!/bin/bash
# 雙擊這個檔案就會啟動 yt2trumpet 的網頁介面並自動打開瀏覽器。
# 第一次使用請先執行 install_mac.sh。
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "還沒安裝，請先執行：bash install_mac.sh"
  read -r -p "按 Enter 關閉…" _
  exit 1
fi
source .venv/bin/activate
echo "啟動 yt2trumpet 介面… 關閉這個終端機視窗就會停止服務。"
yt2trumpet-ui
