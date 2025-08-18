#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Codegen CLI (через собственный API /generate)

Читает промпты из promts/ и пишет код в code/*.py, но вместо прямого вызова OpenAI
отправляет JSON на CODEGEN_API_URL/generate (тот же API, что использует Telegram-бот).

Примеры:
  python ai_codegen.py                                 # обработать все файлы в promts/
  python ai_codegen.py --file promts/x.txt             # обработать один файл
  python ai_codegen.py --model gpt-5 --ext .py         # переопределить модель/расширение
  python ai_codegen.py --env .env                      # указать .env с CODEGEN_API_URL и дефолтами
"""
from __future__ import annotations
import argparse, os, sys, time, re, json
from pathlib import Path
from typing import Optional, Tuple, Dict

try:
    import requests
except ImportError:
    print("Установи пакет: pip install requests", file=sys.stderr); sys.exit(1)

try:
    from dotenv import dotenv_values
except ImportError:
    print("Установи пакет: pip install python-dotenv", file=sys.stderr); sys.exit(1)

DEFAULT_MODEL = "gpt-5"
DEFAULT_PROMPTS_DIR = "promts"   # есть fallback на 'prompts'
DEFAULT_OUTPUT_DIR = "code"
DEFAULT_EXT = ".py"
DEFAULT_ENV_PATH = ".env"

EXT_LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".json": "json", ".md": "markdown", ".html": "html",
    ".css": "css", ".sql": "sql", ".sh": "bash",
    ".yaml": "yaml", ".yml": "yaml",
    ".java": "java", ".go": "go", ".cs": "csharp",
    ".cpp": "cpp", ".rs": "rust", ".kt": "kotlin"
}

FENCE_RE = re.compile(r"```[a-zA-Z0-9_+\-]*\n([\s\S]*?)```", re.M)

def extract_code_block(text: str) -> str:
    """Берём первый fenced-блок ```...``` если есть, иначе весь текст (на всякий случай)."""
    m = FENCE_RE.search(text or "")
    return (m.group(1) if m else (text or "")).strip()

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def resolve_prompts_dir(name: str) -> Path:
    p = Path(name)
    if not p.exists() and name.lower() == "promts":
        alt = Path("prompts")
        if alt.exists():
            return alt
    return p

def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._/-]+", "-", s).strip("-") or "default"

def with_dot(ext: str) -> str:
    return ext if ext.startswith(".") else "." + ext

def make_session_id_from_path(name: str) -> str:
    p = Path(name)
    stem_path = str(p.with_suffix("")).replace(os.sep, "-")
    ext = (p.suffix or ".txt").lower().lstrip(".")
    return sanitize(f"{stem_path}-{ext}")

def detect_lang_by_ext(ext: str, fallback: str = "python") -> str:
    return EXT_LANG.get(ext.lower(), fallback)

def load_env(env_path: Path) -> Dict[str, str]:
    if not env_path.exists():
        raise FileNotFoundError(f"Файл окружения не найден: {env_path.resolve()}")
    cfg = dotenv_values(str(env_path))
    return {k: (v or "").strip() for k, v in (cfg or {}).items()}

def local_latest_path(out_dir: Path, session_id: str, ext: str) -> Path:
    return out_dir / f"{session_id}-latest{ext}"

def next_version_path(out_dir: Path, session_id: str, ext: str) -> Path:
    ensure_dir(out_dir)
    ts = time.strftime("%Y%m%d-%H%M%S")
    return out_dir / f"{session_id}-{ts}{ext}"

def call_codegen_api(base_url: str, payload: dict, timeout: int = 180) -> dict:
    url = base_url.rstrip("/") + "/generate"
    r = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"API {r.status_code}: {r.text}")
    return r.json()

def build_payload(prompt_text: str, *, model: str, session_id: str, ext: str, language: str, base_code: Optional[str]) -> dict:
    mode = "edit" if (base_code and base_code.strip()) else "create"
    return {
        "prompt": prompt_text,
        "base_code": base_code,
        "mode": mode,
        "language": language,
        "model": model,
        "session_id": session_id,
        "ext": ext
    }

def process_prompt_file(
    base_url: str,
    model: str,
    prompt_path: Path,
    out_dir: Path,
    ext: str,
    language: Optional[str] = None
) -> Path:
    spec = prompt_path.read_text(encoding="utf-8").strip()
    if not spec:
        raise ValueError(f"Пустой промпт: {prompt_path}")

    session_id = make_session_id_from_path(prompt_path.stem + ext)  # связываем имя промпта с сессией
    ensure_dir(out_dir)

    latest = local_latest_path(out_dir, session_id, ext)
    base_code = latest.read_text(encoding="utf-8") if latest.exists() else None

    lang = language or detect_lang_by_ext(ext, "python")
    payload = build_payload(spec, model=model, session_id=session_id, ext=ext, language=lang, base_code=base_code)

    data = call_codegen_api(base_url, payload)
    code = data.get("code", "")
    if not code.strip():
        # подстрахуемся, если сервер вернул текст без fenced-блока
        code = extract_code_block(data.get("code", "")) or extract_code_block(str(data))

    # сохраняем как v (по времени) и обновляем latest
    version_path = next_version_path(out_dir, session_id, ext)
    version_path.write_text(code, encoding="utf-8")
    latest.write_text(code, encoding="utf-8")

    print(f"✅ {prompt_path.name} → {version_path.name}  (session={session_id}, mode={'edit' if base_code else 'create'})")
    return version_path

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="AI code generator via your /generate API")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Модель (по умолчанию {DEFAULT_MODEL})")
    ap.add_argument("--prompts-dir", default=DEFAULT_PROMPTS_DIR, help="Папка с промптами (по умолчанию 'promts', fallback 'prompts')")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Папка для вывода версий и latest (по умолчанию 'code')")
    ap.add_argument("--ext", default=DEFAULT_EXT, help="Расширение генерируемых файлов (по умолчанию .py)")
    ap.add_argument("--file", help="Сгенерировать только из одного файла промпта")
    ap.add_argument("--env", default=DEFAULT_ENV_PATH, help="Путь к .env (по умолчанию .env)")
    ap.add_argument("--language", help="Подсказка языка для LLM (если не указан — выбирается по расширению)")
    args = ap.parse_args(argv)

    # .env: берём URL API и дефолты
    try:
        cfg = load_env(Path(args.env))
    except Exception as e:
        print(f"Ошибка загрузки .env: {e}", file=sys.stderr)
        return 1

    base_url = (cfg.get("CODEGEN_API_URL") or "").strip()
    if not base_url:
        print("В .env отсутствует CODEGEN_API_URL (например, https://codegen-railway-production.up.railway.app)", file=sys.stderr)
        return 1

    model = args.model or (cfg.get("DEFAULT_MODEL") or DEFAULT_MODEL)
    out_dir = Path(args.output_dir)
    ext = with_dot(args.ext or (cfg.get("DEFAULT_EXT") or DEFAULT_EXT))
    language = args.language or (cfg.get("DEFAULT_LANGUAGE") or detect_lang_by_ext(ext, "python"))

    processed = 0
    prompts_dir = resolve_prompts_dir(args.prompts_dir)

    try:
        if args.file:
            p = Path(args.file)
            if not p.exists():
                print(f"Файл не найден: {p}", file=sys.stderr); return 1
            process_prompt_file(base_url, model, p, out_dir, ext, language=language)
            processed += 1
        else:
            if not prompts_dir.exists():
                print(f"Папка промптов не найдена: {prompts_dir}", file=sys.stderr); return 1
            patterns = ("*.txt", "*.md", "*.prompt", "*.spec")
            any_found = False
            for pat in patterns:
                for f in prompts_dir.glob(pat):
                    any_found = True
                    try:
                        process_prompt_file(base_url, model, f, out_dir, ext, language=language)
                        processed += 1
                    except Exception as e:
                        print(f"Ошибка {f}: {e}", file=sys.stderr)
            if not any_found:
                print(f"Файлы промптов не найдены в {prompts_dir} (ищу: {', '.join(patterns)})")
    except requests.exceptions.ConnectionError as e:
        print(f"Ошибка подключения к API: {e}", file=sys.stderr); return 1
    except requests.exceptions.ReadTimeout:
        print("Время ожидания ответа API истекло (timeout).", file=sys.stderr); return 1
    except Exception as e:
        print(f"Неожиданная ошибка: {e}", file=sys.stderr); return 1

    print(f"Готово. Сгенерировано: {processed}.")
    return 0 if processed else 2

if __name__ == "__main__":
    raise SystemExit(main())
