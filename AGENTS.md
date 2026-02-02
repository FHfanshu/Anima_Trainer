# Agent Guide for Anima_Trainer

This file is for coding agents working in this repository.
Follow existing project conventions; do not invent new patterns.

## Repo Overview
- Python training pipeline at repo root (`train.py`, `utils/`)
- GUI app under `gui/`
  - Backend: FastAPI in `gui/backend/`
  - Frontend: Vue 3 + Vite in `gui/frontend/`

## Build / Lint / Test Commands

### GUI (backend + frontend)
- One-click launcher with dependency checks:
  - `python start_gui.py`
- Backend only (prod-like):
  - `python gui/run_gui.py`
- Backend dev (hot reload):
  - `python gui/run_gui.py --dev`
- GUI launcher that builds frontend if needed:
  - `python gui/launch.py`

### Frontend (Vue)
- Install deps:
  - `npm install` (run in `gui/frontend`)
- Dev server:
  - `npm run dev`
- Production build:
  - `npm run build`
- Preview build:
  - `npm run preview`
- Typecheck (not in scripts; use npx):
  - `npx vue-tsc --noEmit`

### Python training script
- Install deps:
  - `pip install -r requirements.txt`
- Run training (example):
  - `python train.py --config_file=config/train_config.yaml`
- Accelerate launch example:
  - `accelerate launch train.py --config_file=config/train_config.yaml`

### Tests / Lint
- No test or lint runner is configured in this repo.
- If you add tests in the future, a typical single-test command is:
  - `python -m pytest path/to/test_file.py::TestClass::test_name`
- If you add a JS test runner, document the single-test command here.

## Code Style and Conventions

### General
- Prefer existing patterns over introducing new abstractions.
- Keep changes scoped; do not refactor unrelated code.
- Avoid non-ASCII unless the file already uses it.

### Python (backend + training)
- Style: PEP 8, 4-space indentation, explicit imports.
- Types: use type hints where already used; prefer `Dict[str, Any]` for dynamic data.
- Paths: use `pathlib.Path` for filesystem operations.
- Logging: use `gui/backend/log.py` logger; include context in messages.
- Errors:
  - In FastAPI routes, catch exceptions, log, and raise `HTTPException` with a clear `detail`.
  - Return typed response models (`BaseModel`) where already used.
- API responses:
  - Most GUI APIs return `{status, message, data}` via `SystemResponse` or `TrainResponse`.
  - Keep keys stable; front-end expects `data` nesting.
- Avoid global side effects at import time except for constants.

### FastAPI routing
- Routers live in `gui/backend/api/` and are mounted in `gui/backend/app.py`.
- Use explicit prefixes (`/api/...`) as defined in `app.py`.
- Backwards-compat endpoints may exist in `app.py`; document when added.

### Vue + TypeScript (frontend)
- Use Vue 3 `<script setup lang="ts">` and Composition API.
- Use Pinia stores (`gui/frontend/src/stores/`) for shared state.
- API access goes through `gui/frontend/src/api/client.ts`.
- Response handling:
  - Axios client returns `response.data` (already unwrapped by interceptor).
  - Backend wraps payloads under `data`; read from `response.data`.
- Types:
  - Use `type` imports (`import type { ... }`).
  - Keep API DTO fields aligned with backend keys (snake_case).
- Imports:
  - Order: external libs, internal modules, then types.
  - Keep grouped and minimal.
- Naming:
  - `camelCase` for variables/functions, `PascalCase` for Vue components.
  - Filenames follow existing conventions (`HomeView.vue`, `TrainView.vue`).

### UI / UX
- Keep Element Plus components consistent with existing layouts.
- Avoid introducing new UI patterns without matching existing styles.
- Ensure views work on desktop and mobile.

## API Expectations (current)
- Backend system info: `/api/system/info` returns metadata and optional versions.
- GPU status: `/api/system/gpu_status` returns `{gpus: [...]}`.
- Checkpoints list: `/api/train/checkpoints` returns `{checkpoints: [...]}`.
- Config endpoints: `/api/config/*` (default, presets, load, save, delete).

## Dependency Notes
- GUI launcher (`start_gui.py`) installs only GUI runtime deps.
- Full training stack requires `requirements.txt`.
- `python-multipart` is needed for FastAPI multipart uploads.

## Cursor/Copilot Rules
- No `.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md` found.
- If any are added later, update this file with their requirements.

## When You Add/Change Commands
- Update this file to keep build/test instructions accurate.
- Prefer concrete commands over "run X" phrasing.
