#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import difflib
from pathlib import Path
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import dotenv_values

try:
    from openai import OpenAI
except ImportError as e:
    raise SystemExit("Установи зависимости: pip install -r requirements.txt") from e

# ─── .env ONLY ────────────────────────────────────────────────────────────
def load_env(path: Path = Path(".env")) -> dict:
    if not path.exists():
        raise RuntimeError(f"Файл окружения не найден: {path.resolve()}")
    cfg = dotenv_values(str(path))
    api_key = (cfg.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(f"В {path.resolve()} отсутствует OPENAI_API_KEY")
    return cfg

CFG = load_env()
DEFAULT_MODEL = (CFG.get("DEFAULT_MODEL") or "gpt-5").strip()
SESSIONS_DIR = Path(CFG.get("SESSIONS_DIR") or "sessions"); SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR   = Path(CFG.get("OUTPUT_DIR") or "code");     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLIENT = OpenAI(
    api_key=(CFG.get("OPENAI_API_KEY") or "").strip(),
    base_url=(CFG.get("OPENAI_BASE_URL") or "").strip() or None
)

# ─── Helpers ─────────────────────────────────────────────────────────────
FENCE_RE = re.compile(r"```[a-zA-Z0-9_+\-]*\n([\s\S]*?)```", re.M)

def extract_code_block(text: str) -> str:
    m = FENCE_RE.search(text)
    return (m.group(1) if m else text).strip()

def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._/-]+", "-", s).strip("-") or "default"

def ext_with_dot(ext: str) -> str:
    return ext if ext.startswith(".") else f".{ext}"

def next_version_path(session_dir: Path, ext: str) -> Path:
    session_dir.mkdir(parents=True, exist_ok=True)
    extsuf = ext.lstrip(".")
    prev = sorted(session_dir.glob(f"v*.{extsuf}"))
    n = 1
    if prev:
        try:
            n = int(prev[-1].stem[1:]) + 1
        except Exception:
            n = len(prev) + 1
    return session_dir / f"v{n:03d}{ext}"

SYSTEM = (
    "You are a strict, production-grade code generator.\n"
    "Return only the COMPLETE UPDATED CODE in a single fenced block. No explanations."
)

CREATE_TPL = """Language: {language}
Task: Generate a single, production-ready code file strictly matching the specification.
Rules:
- Return ONLY code in one fenced block ```...```.
- Deterministic, self-contained; no external secrets.
- If details are missing, choose sensible production defaults.
Specification:
{spec}
"""

EDIT_TPL = """Language: {language}
You will receive:
1) The CURRENT code of a single file.
2) A CHANGE REQUEST describing modifications to apply.
Goal:
- Produce the FULL UPDATED FILE implementing the request.
- Preserve working logic unless the request requires changes.
- Keep imports, structure, and style consistent where possible.
Rules:
- Return ONLY the updated code in one fenced block ```...```.
- Deterministic, production-grade; no comments/prose.
CHANGE REQUEST:
{spec}

CURRENT CODE:
```
{current_code}
```
"""

Mode = Literal["create", "edit"]

class GenerateRequest(BaseModel):
    prompt: str
    base_code: Optional[str] = None
    mode: Mode = "create"
    language: str = "python"
    model: str = DEFAULT_MODEL
    session_id: str = "default"
    ext: str = ".py"

class GenerateResponse(BaseModel):
    session_id: str
    version_path: str
    mirrored_output_path: str
    code: str
    diff: Optional[str] = None
    saved: bool = True

app = FastAPI(title="Iterative Codegen API", version="1.1")

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    session = sanitize(req.session_id)
    session_dir = SESSIONS_DIR / session
    ext = ext_with_dot(req.ext)

    if req.mode == "edit":
        if not req.base_code:
            raise HTTPException(status_code=400, detail="Для режима 'edit' нужно поле base_code.")
        template = EDIT_TPL.format(language=req.language, spec=req.prompt, current_code=req.base_code)
    else:
        template = CREATE_TPL.format(language=req.language, spec=req.prompt)

    try:
        resp = CLIENT.responses.create(model=req.model, instructions=SYSTEM, input=template)
        text = getattr(resp, "output_text", None) or str(resp)
        code = extract_code_block(text)
        if not code.strip():
            raise RuntimeError("Пустой ответ модели")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {e}")

    version_path = next_version_path(session_dir, ext)
    version_path.write_text(code, encoding="utf-8")

    latest = OUTPUT_DIR / f"{session}-latest{ext}"
    latest.write_text(code, encoding="utf-8")

    udiff = None
    if req.base_code:
        udiff = "\n".join(difflib.unified_diff(
            req.base_code.splitlines(), code.splitlines(),
            fromfile=f"before{ext}", tofile=f"after{ext}", lineterm=""
        ))

    return GenerateResponse(
        session_id=session,
        version_path=str(version_path),
        mirrored_output_path=str(latest),
        code=code,
        diff=udiff,
        saved=True
    )
