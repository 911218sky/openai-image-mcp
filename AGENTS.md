# oepnai_image Agent Guide

## 先做什麼

- 先看 `README.md`
- 要改流程時，先看 `src/oepnai_image/cli.py`
- 有回滾風險時，先做 Git 狀態檢查再改

## 工作原則

- 優先沿用現有結構：`src/`、`batches/`、`prompt_styles/`
- 不要提交 `.env`、生成圖片、batch 輸出、快取、大型 artifacts
- 若任務涉及 OpenAI API 最新行為，先查官方文件

## 驗證

```bash
rtk uv run pytest -q
rtk python -m compileall -q src
```
