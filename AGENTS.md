# oepnai_image Agent Guide

## 預設流程

改檔、調整 prompt、修改批次生成流程、整理文件或準備回滾前，先做 Git 現況檢查，再做小步修改；重要節點要建立 checkpoint commit，方便後續回滾。

如有回滾風險，先讀 `/home/sbplab/sky/agents/.codex/skills/rollback-control/SKILL.md`。

## 工作原則

- 優先使用既有 `src/`、`batches/`、`prompt_styles/` 結構。
- 修改生成流程前，先讀 `README.md` 與目前 CLI 入口。
- 不要把 `.env`、生成圖片、批次輸出、快取或大型 artifact 提交進 Git。
- 若任務涉及 OpenAI API 最新用法，先查官方文件再改。

## 常用驗證

```bash
rtk uv run python -m pytest -q
rtk python -m compileall -q src
```
