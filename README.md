# oepnai_image

用 CLI 產生 OpenAI 圖片，支援：

- 單一或多個 prompt
- 從文字檔載入 prompts
- 從 JSON batch 跑多筆 jobs
- style 模板
- 自訂尺寸、比例、輸出路徑

## 快速開始

```bash
cd /home/sbplab/sky/agents/gen_image
cp .env.example .env
uv sync
```

`.env` 最少要填：

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_IMAGE_MODEL=gpt-image-1
OPENAI_TIMEOUT_SECONDS=1200
```

注意：

- `OPENAI_BASE_URL` 要填 API 根路徑，通常要到 `.../v1`
- 尺寸的寬高都必須可被 `16` 整除
- 如果供應商回 `Image generation is not enabled for this group`，通常是 key/權限問題，不是 prompt 問題

## 最常用指令

顯示說明：

```bash
uv run openai-image --help
```

單張：

```bash
uv run openai-image --prompt "a cinematic photo of a red sports car"
```

多張 prompt：

```bash
uv run openai-image \
  --prompt "a watercolor cat reading a newspaper" \
  --prompt "an isometric tiny bookstore at night"
```

每個 prompt 生成多張：

```bash
uv run openai-image \
  --prompt "a minimalist mountain logo" \
  --num-images 3
```

指定輸出資料夾與檔名前綴：

```bash
uv run openai-image \
  --prompt "a cinematic photo of a red sports car" \
  --output-dir ./outputs/car \
  --filename-prefix car-hero
```

只看最終 payload，不打 API：

```bash
uv run openai-image \
  --prompt "a flat illustration of a robot barista" \
  --dry-run
```

## 尺寸

直接指定：

```bash
uv run openai-image \
  --prompt "a futuristic city skyline at dusk" \
  --size 1536x1024
```

或用寬高：

```bash
uv run openai-image \
  --prompt "a futuristic city skyline at dusk" \
  --width 1536 \
  --height 1024
```

或用比例：

```bash
uv run openai-image \
  --prompt "a futuristic city skyline at dusk" \
  --aspect-ratio 16:9
```

可再搭配：

```bash
uv run openai-image \
  --prompt "a futuristic city skyline at dusk" \
  --aspect-ratio 16:9 \
  --long-edge 1920
```

避免這種非法尺寸：

```text
1320x1080
```

如果你要 `11:9` 橫幅，優先用：

- `1232x1008`
- `1408x1152`
- `1584x1296`

## Prompt 檔與 Batch

文字檔每行一個 prompt：

```bash
uv run openai-image --prompts-file prompts.txt
```

`prompts.txt`：

```text
a studio product photo of a glass bottle
an anime street scene in the rain
```

Batch：

```bash
uv run openai-image --batch jobs.json
```

最小格式：

```json
{
  "defaults": {
    "category": "misc",
    "model": "gpt-image-1",
    "size": "1024x1024",
    "background": "opaque",
    "n": 1
  },
  "jobs": [
    {
      "slug": "poster-concept",
      "prompt": "a bold movie poster for a sci-fi thriller"
    }
  ]
}
```

## Style

列出 styles：

```bash
uv run openai-image --list-styles
```

套用 style：

```bash
uv run openai-image \
  --style paper-figure \
  --prompt "a diagram explaining microphone-array beamforming"
```

內建 `paper-figure` 適合論文圖草稿，預設會帶：

- `model=gpt-image-2`
- `size=1536x864`
- `quality=medium`

style 檔放在 `prompt_styles/`，格式例如：

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

## 輸出

預設輸出到：

```text
generated_images/<timestamp>/
```

每次執行會寫出：

- 生成的圖片
- `manifest.json`

## 測試

```bash
uv sync --extra dev
uv run pytest -q
```
