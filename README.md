# oepnai_image

用 OpenAI 圖像模型從 CLI 產生圖片，支援：

- 直接輸入單一 prompt
- 一次輸入多個 prompts
- 從文字檔讀入多個 prompts
- 從 JSON batch 檔執行多筆 job
- 每個 prompt 一次生成多張圖片
- 從 prompt style 資料夾載入可重用風格模板
- 指定輸出資料夾
- 指定檔名前綴
- 平鋪輸出，不分子資料夾
- 指定解析度
- 用高寬比自動換算尺寸

## Requirements

- `uv`
- Python `3.12`
- `OPENAI_API_KEY`

## Setup

```bash
cd oepnai_image
cp .env.example .env
uv sync
```

如果要跑測試：

```bash
uv sync --extra dev
```

## Environment

`.env`:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_IMAGE_MODEL=gpt-image-1
OPENAI_TIMEOUT_SECONDS=180
```

`OPENAI_BASE_URL` 要填 OpenAI 相容 API 根路徑，不是供應商首頁。例如很多代理需要填到 `.../v1`，像 `https://api.tu-zi.com/coding/v1`。

`OPENAI_TIMEOUT_SECONDS` 是單次 API 請求與圖片下載的等待上限。供應商卡住時，CLI 會在超時後進入重試或報錯，不會無限等下去。

如果 API 回：

```text
Image generation is not enabled for this group
```

這通常表示供應商那邊的 API key 或群組沒有開通圖片生成功能，不是 prompt、尺寸或 CLI 參數錯誤。

## Usage

顯示說明：

```bash
uv run openai-image --help
```

單一 prompt：

```bash
uv run openai-image --prompt "a cinematic photo of a red sports car"
```

多個 prompts：

```bash
uv run openai-image \
  --prompt "a watercolor cat reading a newspaper" \
  --prompt "an isometric tiny bookstore at night" \
  --output-dir ./outputs/batch-01
```

每個 prompt 生成多張：

```bash
uv run openai-image \
  --prompt "a minimalist mountain logo" \
  --num-images 3
```

### 預設草稿策略：同一概念畫三張

做論文概念圖或方法圖時，預設不要只用同一個 prompt 搭配 `--num-images 3`。建議改成三個不同提示、三種構圖角度，並用 `--workers 3` 並行生成三張草稿：

- prompt 1：主流程或架構 zoom-in
- prompt 2：機制或公式概念
- prompt 3：輸出曲線、before/after 或直覺比較

這樣三張圖會真的提供不同設計方向，而不是同一張圖的隨機變體。常用指令：

```bash
uv run openai-image \
  --batch batches/my-paper-figure-3drafts.json \
  --output-dir generated_images/my-paper-figure-3drafts \
  --workers 3 \
  --quality medium \
  --size 1536x864
```

指定解析度：

```bash
uv run openai-image \
  --prompt "a futuristic city skyline at dusk" \
  --resolution 1536x1024
```

或用寬高直接指定：

```bash
uv run openai-image \
  --prompt "a futuristic city skyline at dusk" \
  --width 1536 \
  --height 1024
```

指定高寬比：

```bash
uv run openai-image \
  --prompt "a futuristic city skyline at dusk" \
  --aspect-ratio 16:9
```

搭配長邊或短邊控制實際輸出尺寸：

```bash
uv run openai-image \
  --prompt "a futuristic city skyline at dusk" \
  --aspect-ratio 16:9 \
  --long-edge 1920
```

```bash
uv run openai-image \
  --prompt "a book cover illustration" \
  --aspect-ratio 3:4 \
  --short-edge 1024
```

指定檔名前綴：

```bash
uv run openai-image \
  --prompt "a cinematic photo of a red sports car" \
  --filename-prefix car-hero \
  --output-dir ./outputs/car
```

多個 prompts 搭配自訂前綴時，檔名會自動變成：

```text
car-hero-01.png
car-hero-02.png
```

平鋪輸出，不建立 category 子資料夾：

```bash
uv run openai-image \
  --prompt "a watercolor cat reading a newspaper" \
  --prompt "an isometric tiny bookstore at night" \
  --output-dir ./outputs/flat \
  --filename-prefix concept \
  --flat-output
```

從文字檔讀多個 prompts：

```bash
uv run openai-image --prompts-file prompts.txt
```

`prompts.txt` 每行一個 prompt：

```text
a studio product photo of a glass bottle
an anime street scene in the rain
```

搭配 batch JSON：

```bash
uv run openai-image --batch jobs.json
```

只檢查會送出的內容，不打 API：

```bash
uv run openai-image \
  --prompt "a flat illustration of a robot barista" \
  --prompt "a paper-cut forest scene" \
  --dry-run
```

列出可用 style：

```bash
uv run openai-image --list-styles
```

套用內建 style：

```bash
uv run openai-image \
  --style paper-figure \
  --prompt "a diagram explaining microphone-array beamforming"
```

`paper-figure` 預設使用 `gpt-image-2`、`1536x864`、`quality=medium`，讓日常草稿和多張 batch 不會每次都卡在 4K high quality。需要最後輸出 4K 論文圖時，再明確覆蓋：

```bash
uv run openai-image \
  --style paper-figure \
  --prompt "a diagram explaining microphone-array beamforming" \
  --size 3840x2160 \
  --quality high
```

若相容 API 實際回傳的尺寸和指定尺寸不同，CLI 只會原樣寫出圖片，不會再重採樣；實際尺寸會記錄在 `manifest.json`。這個 style 預設要求單張圖只聚焦一個主要概念，避免把資料流、模型、訓練、指標全部塞在同一張圖。

指定自訂 style 資料夾：

```bash
uv run openai-image \
  --style-dir ./prompt_styles \
  --style paper-figure \
  --prompt "a concept figure for local directional coding"
```

## Output

預設會輸出到：

```text
generated_images/<timestamp>/
```

也可以用 `--output-dir` 指定目錄。

OpenAI 官方文件目前明確提到可以自訂 `size`，範例包含 `1024x1024`、`1024x1536`；最新圖片模型文件也提到 `gpt-image-2` 支援很多有效解析度，所以這個 CLI 直接把你指定的尺寸送給 API。

每個 job 會寫出：

- 對應圖片檔
- `manifest.json`

## Prompt Styles

內建 style 會放在：

```text
prompt_styles/
```

目前預設附一個：

- `paper-figure`: 參考論文圖解風格，白底、簡潔向量圖、藍綠橘點綴、箭頭與標註清楚、適合學術論文配圖；預設 `gpt-image-2`、`1536x864`、`quality=medium`；單張圖建議只放 3 到 5 個主要視覺區塊

每個 style 檔都是 JSON，例如：

```json
{
  "slug": "paper-figure",
  "name": "Paper Figure",
  "description": "Clean academic schematic figures.",
  "template": "Create a clean academic figure about: {prompt}",
  "defaults": {
    "category": "paper-figures",
    "model": "gpt-image-2",
    "size": "1536x864",
    "background": "opaque",
    "quality": "medium"
  }
}
```

`template` 裡的 `{prompt}` 會被你的主題 prompt 自動替換。

## Focused Paper Figure Batches

複雜研究流程建議拆成多張圖，每張只講一個面向。AudioScan 預設模型已提供一份 batch：

```bash
uv run openai-image \
  --batch batches/audioscan-default-model-focused.json \
  --output-dir generated_images/audioscan-default-model-focused
```

這份 batch 會分別生成：

- data to spectrogram
- encoder backbone
- projection head
- FC regression head
- two-stage training
- target aggregation

這種拆法通常比單張大總覽更穩定，AI 也比較不會把文字和 layer 細節畫錯。

若是同一個概念要挑論文圖草稿，建議用 3-job batch，每個 job 都圍繞同一核心概念，但提示詞要刻意改變視角。例如：

```json
{
  "defaults": {
    "category": "paper-figures",
    "style": "paper-figure",
    "n": 1
  },
  "jobs": [
    {
      "slug": "concept-zoom",
      "filename_prefix": "concept-01-zoom",
      "prompt": "Show the shared model pipeline once, then zoom into the key module where the method differs."
    },
    {
      "slug": "concept-mechanism",
      "filename_prefix": "concept-02-mechanism",
      "prompt": "Show the same method as a compact mechanism or equation-style concept figure."
    },
    {
      "slug": "concept-effect",
      "filename_prefix": "concept-03-effect",
      "prompt": "Show the same method through the before/after effect on output curves or predictions."
    }
  ]
}
```

## Batch JSON Format

```json
{
  "defaults": {
    "category": "misc",
    "model": "gpt-image-1",
    "size": "1024x1024",
    "background": "opaque",
    "n": 1,
    "style": "paper-figure"
  },
  "jobs": [
    {
      "slug": "poster-concept",
      "prompt": "a bold movie poster for a sci-fi thriller"
    },
    {
      "slug": "logo-set",
      "prompt": "a clean geometric fox logo",
      "category": "branding",
      "n": 3
    }
  ]
}
```
