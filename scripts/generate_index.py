#!/usr/bin/env python3
"""掃描 reports/ 目錄，產生 index.html 首頁索引（最新日期在最上面）"""

import re
from pathlib import Path

REPORTS_DIR = Path(__file__).parent.parent / 'reports'
OUTPUT = Path(__file__).parent.parent / 'index.html'

DATE_RE = re.compile(r'(\d{4}-\d{2}-\d{2})_report\.html')


def collect_reports():
    """收集所有報告，回傳 [(date_str, filename), ...] 降冪排序"""
    entries = []
    for f in REPORTS_DIR.glob('*_report.html'):
        m = DATE_RE.match(f.name)
        if m:
            entries.append((m.group(1), f.name))
    entries.sort(key=lambda x: x[0], reverse=True)
    return entries


def render_html(entries):
    if entries:
        date_range = f'{entries[-1][0]} ~ {entries[0][0]}'
    else:
        date_range = '尚無報告'

    rows = ''
    for i, (date_str, filename) in enumerate(entries):
        badge = '<span class="badge new">NEW</span>' if i == 0 else ''
        rows += f'''    <a href="reports/{filename}" class="report-link">
      <span class="date">{date_str}</span>
      <span class="title">PTT Stock 整合分析報告</span>
      {badge}
    </a>
'''

    return f'''<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PTT Stock 分析報告索引</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: "Microsoft JhengHei", "PingFang TC", sans-serif;
    background: #0d1117;
    color: #e6edf3;
    min-height: 100vh;
    padding: 40px 20px;
  }}
  .container {{
    max-width: 720px;
    margin: 0 auto;
  }}
  h1 {{
    color: #58a6ff;
    font-size: 1.8em;
    margin-bottom: 4px;
  }}
  .subtitle {{
    color: #8b949e;
    margin-bottom: 8px;
  }}
  .stats {{
    color: #8b949e;
    font-size: 0.9em;
    margin-bottom: 28px;
    padding-bottom: 16px;
    border-bottom: 1px solid #30363d;
  }}
  .report-list {{
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}
  .report-link {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 18px;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    text-decoration: none;
    color: #e6edf3;
    transition: border-color 0.2s, background 0.2s;
  }}
  .report-link:hover {{
    border-color: #58a6ff;
    background: #1c2333;
  }}
  .date {{
    color: #58a6ff;
    font-weight: 600;
    font-size: 1.05em;
    min-width: 120px;
  }}
  .title {{
    color: #8b949e;
    font-size: 0.95em;
  }}
  .badge {{
    font-size: 0.7em;
    padding: 2px 8px;
    border-radius: 12px;
    font-weight: 600;
    margin-left: auto;
  }}
  .badge.new {{
    background: #1b4332;
    color: #3fb950;
  }}
  .empty {{
    color: #8b949e;
    text-align: center;
    padding: 60px 0;
    font-size: 1.1em;
  }}
</style>
</head>
<body>
<div class="container">
  <h1>PTT Stock 分析報告</h1>
  <div class="subtitle">推文語意分析 + 海龜通道技術分析</div>
  <div class="stats">共 {len(entries)} 份報告 | {date_range}</div>
  <div class="report-list">
{rows if rows else '    <div class="empty">尚無報告，請先執行分析並部署。</div>'}
  </div>
</div>
</body>
</html>'''


def main():
    entries = collect_reports()
    html = render_html(entries)
    OUTPUT.write_text(html, encoding='utf-8')
    print(f'index.html generated: {len(entries)} reports')


if __name__ == '__main__':
    main()
