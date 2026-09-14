# yt2trumpet — YouTube 主旋律 → Bb 小號五線譜

把 YouTube 影片（或本機音檔）的主旋律自動扒下來，移調成 **Bb 小號記譜音**，輸出 PDF / PNG 五線譜與可用 MuseScore 編輯的 MusicXML。

```
YouTube 網址 ──yt-dlp──▶ 音訊 ──Demucs──▶ 人聲 / 旋律樂器軌
   ──pYIN 或 Basic Pitch──▶ 音符事件 ──節拍偵測 + 量化──▶ 拍點上的音符
   ──實音 +大二度、音域折疊──▶ 小號記譜音 ──music21──▶ MusicXML ──MuseScore / verovio──▶ PDF、PNG
```

> 自動扒譜不會 100% 正確。設計目標是「拿到一份八九成對、可以直接在 MuseScore 裡修的譜」，
> 而不是完全免修。每次輸出都會附 `.musicxml`，請把它丟進 MuseScore 校正。

## 安裝

需要 Python 3.10 以上。

```bash
cd tools/yt2trumpet

# 基本（純 pip，不需要深度學習框架；可處理獨奏、人聲清楚的音檔）
pip install -e .

# 建議：加裝人聲分離（流行歌幾乎都需要）。會拉 PyTorch，約 2 GB
pip install -e ".[separate]"

# 選配：多音旋律抽取（純音樂 / 動漫 OST / 器樂混音時明顯較準）。會拉 TensorFlow
pip install -e ".[polyphonic]"

# 全部
pip install -e ".[all]"
```

其他工具：

| 工具 | 是否必要 | 說明 |
| --- | --- | --- |
| ffmpeg | 建議 | 沒裝也能跑，會自動改用 `imageio-ffmpeg` 內附的執行檔 |
| MuseScore 3/4 | 選配 | 有裝的話 PDF/PNG 會用它排版（品質最好）；沒裝就用 `verovio` 排版 |
| cairo | verovio 路線需要 | Linux：`apt install libcairo2`；macOS：`brew install cairo`；Windows：`pip install cairosvg` 通常會附 |

第一次跑 Demucs 會自動下載約 80 MB 的模型。

## 使用

```bash
# 最簡單：貼網址
yt2trumpet "https://www.youtube.com/watch?v=xxxxxxxxxxx"

# 只扒副歌（第 62 秒到 95 秒），旋律是人聲
yt2trumpet "https://youtu.be/xxxx" --start 62 --end 95 --stem vocals

# 本機音檔、初學者音域、自動挑好吹的調
yt2trumpet song.mp3 --range beginner --transpose auto

# 自己指定速度與第一個強拍的位置（自動偵測不準時）
yt2trumpet song.mp3 --bpm 96 --offset 0.35

# 3/4 拍、八分音符為最小單位
yt2trumpet waltz.mp3 --time 3/4 --grid 2
```

輸出在 `output/<標題>/`：

```
output/歌名/
├── 歌名.musicxml   ← 丟進 MuseScore 修正
├── 歌名.pdf
├── 歌名-1.png      ← 每頁一張
└── 歌名-1.svg      （verovio 路線才有）
```

跑完會印出摘要：速度、實音調性、小號記譜調性、旋律來自哪一軌、音域折疊了幾個音。

### 主要參數

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `--start` / `--end` | 整首 | 裁切秒數 |
| `--stem` | `auto` | 旋律來源：`auto` 依人聲能量占比自動選 / `vocals` 人聲 / `other` 旋律樂器 / `none` 不分離 |
| `--pitch` | `auto` | `pyin`（單音，人聲軌預設）/ `basic-pitch`（多音，器樂預設，需安裝） |
| `--range` | `intermediate` | 記譜音域：`beginner` C4–G5、`intermediate` G3–C6、`advanced` F#3–E6，或自訂 `G3-C6` |
| `--transpose` | `0` | 整首移調半音數。`0` 保留原調；`auto` 在 ±6 半音內挑「不超音域、調號最少、移最少」的調 |
| `--bpm` / `--offset` | 自動偵測 | 手動固定速度與第一個強拍的秒數 |
| `--downbeat` | 自動猜 | 自動抓拍時，第幾個拍點（0～拍數-1）是第一個強拍。小節線歪一拍時用這個修 |
| `--time` | `4/4` | 拍號 |
| `--grid` | `4` | 每拍最小等分：4 = 十六分音符，2 = 八分音符，3 = 三連音 |
| `--min-note` | `70` | 最短音符（毫秒），更短視為雜訊 |
| `--max-rest-bars` | `2` | 前奏、間奏超過這麼多小節會被壓縮，避免整頁休止符 |
| `--formats` | `pdf,png` | MusicXML 永遠會輸出 |
| `--renderer` | `auto` | `musescore` / `verovio`，`auto` 有 MuseScore 就用它 |
| `--device` | 自動 | Demucs 用 `cpu` / `cuda` / `mps` |
| `--keep-temp` | 關 | 把下載的音訊與分離後的四軌留在 `output/<標題>/work/` |

### 小號相關規則

- Bb 小號是移調樂器：**記譜音 = 實音 + 大二度**。實音 C 大調的歌，譜上會是 D 大調（兩個升號）。
- 中級音域預設 G3～C6（記譜）。超出的音先以**整個樂句**為單位折八度（保留旋律輪廓），
  整句折了還不行才逐音折。摘要會告訴你折了幾個音。
- 預設**保留原調**（跟原曲一起吹才對得上）。想要好吹的調就 `--transpose auto`，或直接給半音數。
- 降記號調（F、Bb、Eb…）會用降記號拼法，不會出現一堆 A#。

## 結果不理想時

| 現象 | 試試看 |
| --- | --- |
| 譜上抓到伴奏、和聲，不是主旋律 | `--stem vocals`（唱歌）或 `--stem other`（器樂）；只裁旋律清楚的段落 `--start/--end` |
| 音符碎成一堆十六分音符 | `--grid 2`、`--min-note 120` |
| 速度抓成兩倍 / 一半 | `--bpm` 直接指定 |
| 小節線位置錯一拍 | `--downbeat 1`（或 2、3）試到對為止；或改用 `--bpm` + `--offset` |
| 高八度 / 低八度不對 | `--range advanced` 放寬，或 `--transpose` 手動 |
| 純音樂抓不到旋律 | 安裝 `[polyphonic]` 後用 `--pitch basic-pitch` |
| 中文標題在 PDF 變空白 | 系統缺中文字型；設環境變數 `YT2TRUMPET_FONT="Microsoft JhengHei"`（或 `PingFang TC`） |
| 沒有 GPU 太慢 | Demucs 4 分鐘的歌 CPU 約 3～8 分鐘；先用 `--start/--end` 只切需要的段落 |

## 專案結構

```
yt2trumpet/
├── audio.py      下載（yt-dlp）、ffmpeg 轉檔、裁切
├── separate.py   Demucs 分離、人聲占比判斷
├── pitch.py      pYIN / Basic Pitch 音高追蹤 → 音符事件；多音→單旋律的天際線啟發式
├── rhythm.py     節拍偵測、量化到拍點格線、強拍推測、長休止壓縮
├── trumpet.py    實音→記譜音、音域折疊、Krumhansl 調性偵測、自動移調
├── score.py      music21 建譜 → MusicXML；MuseScore / verovio 渲染；中文字型處理
├── pipeline.py   串接
└── cli.py        命令列
tests/            pytest（用合成的原創旋律做端到端驗證）
```

測試：`pip install -e ".[dev]" && pytest`

## 限制與注意事項

- 只支援單一旋律線；和聲、對位、鼓不會出現在譜上。
- 節奏量化假設速度大致固定。自由速度（rubato）的段落請自己給 `--bpm`。
- 人聲的滑音、轉音會被吸收成同一個音；裝飾音需要自己補。
- Demucs 分離後的人聲軌偶爾會殘留合唱或吉他，會被當成旋律抓進去，需要人工刪。
- 下載 YouTube 內容請遵守 YouTube 服務條款與著作權法，本工具僅供個人練習使用。
