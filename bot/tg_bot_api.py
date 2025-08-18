#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram Codegen Bot (API client)

Возможности:
- Несколько файлов/типов (например: code.py, langgraph.py, web/index.html).
- Версионирование: sessions/<chat>/<session>/vNNN.<ext> + latest.<ext>.
- Команды: /use (или /file), /files, /history, /get, /rollback, /bundle, /model, /lang, /state.
- Для правок можно прислать промпт + код в блоке из трёх обратных кавычек (triple backticks).

Требуется внешний API /generate (FastAPI из каталога api/).
Секреты и URL — ТОЛЬКО из .env (на Railway он собирается в start.sh).
"""
from __future__ import annotations

import os
import json
import re
import time
import difflib
import zipfile
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import requests
from dotenv import dotenv_values
from telegram import Update, InputFile
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# ─────────── .env ONLY ───────────
CFG = dotenv_values(".env")

def need(k: str) -> str:
    v = (CFG.get(k) or "").strip()
    if not v:
        raise SystemExit(f"В .env отсутствует {k}")
    return v

TELEGRAM_TOKEN    = need("TELEGRAM_TOKEN")
CODEGEN_API_URL   = need("CODEGEN_API_URL")
DEFAULT_MODEL     = (CFG.get("DEFAULT_MODEL") or "gpt-5").strip()
DEFAULT_LANGUAGE  = (CFG.get("DEFAULT_LANGUAGE") or "python").strip()
DEFAULT_FILE_NAME = (CFG.get("DEFAULT_FILE_NAME") or "code.py").strip()
SESSIONS_ROOT     = Path(CFG.get("SESSIONS_DIR") or "/data/sessions"); SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT       = Path(CFG.get("OUTPUT_DIR") or "/data/code");       OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ─────────── Авто-язык по расширению ───────────
EXT_LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".json": "json", ".md": "markdown", ".html": "html",
    ".css": "css", ".sql": "sql", ".sh": "bash",
    ".yaml": "yaml", ".yml": "yaml",
    ".java": "java", ".go": "go", ".cs": "csharp",
    ".cpp": "cpp", ".rs": "rust", ".kt": "kotlin"
}

# ─────────── Регэксп кода в блоке из трёх обратных кавычек ───────────
FENCE_RE = re.compile(r"```[a-zA-Z0-9_+\-]*\n([\s\S]*?)```", re.M)

# ─────────── Утилиты ───────────
def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._/-]+", "-", s).strip("-") or "default"

def with_dot(ext: str) -> str:
    return ext if ext.startswith(".") else "." + ext

def file_ext(name: str) -> str:
    ext = Path(name).suffix
    return ext if ext else ".txt"

def make_session_id_from_path(name: str) -> str:
    p = Path(name)
    stem_path = str(p.with_suffix("")).replace(os.sep, "-")
    ext = (p.suffix or ".txt").lower().lstrip(".")
    return sanitize(f"{stem_path}-{ext}")

def parse_msg(text: str) -> Tuple[str, Optional[str]]:
    m = FENCE_RE.search(text or "")
    if not m:
        return (text.strip(), None)
    code = m.group(1).strip()
    prompt = (text[:m.start()] + text[m.end():]).strip()
    return (prompt, code if code else None)

def chat_root(chat_id: int) -> Path:
    d = SESSIONS_ROOT / str(chat_id)
    d.mkdir(parents=True, exist_ok=True)
    return d

def session_dir(chat_id: int, session_id: str) -> Path:
    d = chat_root(chat_id) / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def latest_path(chat_id: int, session_id: str, ext: str) -> Path:
    return session_dir(chat_id, session_id) / f"latest{ext}"

def version_paths(chat_id: int, session_id: str, ext: str) -> List[Path]:
    return sorted(session_dir(chat_id, session_id).glob(f"v*.{ext.lstrip('.')}"))

def next_version_path(chat_id: int, session_id: str, ext: str) -> Path:
    prev = version_paths(chat_id, session_id, ext)
    n = 1
    if prev:
        try:
            n = int(prev[-1].stem[1:]) + 1
        except Exception:
            n = len(prev) + 1
    return session_dir(chat_id, session_id) / f"v{n:03d}{ext}"

# ─────────── Манифест ───────────
def manifest_path(chat_id: int) -> Path:
    return chat_root(chat_id) / "manifest.json"

def load_manifest(chat_id: int) -> dict:
    p = manifest_path(chat_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"files": {}}

def save_manifest(chat_id: int, m: dict) -> None:
    manifest_path(chat_id).write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

def touch_manifest_entry(chat_id: int, file_name: str) -> Tuple[str, str]:
    session_id = make_session_id_from_path(file_name)
    ext = file_ext(file_name)
    m = load_manifest(chat_id)
    ent = m["files"].get(session_id) or {"file_name": file_name, "ext": ext, "last_version": 0, "updated_at": None}
    vlist = version_paths(chat_id, session_id, ext)
    ent["last_version"] = max(ent.get("last_version", 0), len(vlist))
    ent["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    m["files"][session_id] = ent
    save_manifest(chat_id, m)
    return session_id, ext

def register_new_version(chat_id: int, session_id: str, ext: str) -> int:
    m = load_manifest(chat_id)
    ent = m["files"].get(session_id) or {"file_name": f"{session_id}{ext}", "ext": ext, "last_version": 0, "updated_at": None}
    ent["last_version"] = ent.get("last_version", 0) + 1
    ent["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    m["files"][session_id] = ent
    save_manifest(chat_id, m)
    return ent["last_version"]

def list_files_summary(chat_id: int) -> List[str]:
    m = load_manifest(chat_id)
    rows = []
    for sid, ent in sorted(m["files"].items()):
        rows.append(f"• {ent['file_name']}  (session={sid}, ext={ent['ext']}, versions={ent.get('last_version',0)})")
    return rows

# ─────────── User state ───────────
def state_file(chat_id: int) -> Path:
    return OUTPUT_ROOT / f"state-{chat_id}.json"

def load_state(chat_id: int) -> Dict[str, str]:
    p = state_file(chat_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    file_name = DEFAULT_FILE_NAME
    st = {
        "file_name": file_name,
        "session_id": make_session_id_from_path(file_name),
        "model": DEFAULT_MODEL,
        "language": EXT_LANG.get(file_ext(file_name).lower(), DEFAULT_LANGUAGE)
    }
    save_state(chat_id, st)
    return st

def save_state(chat_id: int, st: Dict[str, str]) -> None:
    state_file(chat_id).write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    touch_manifest_entry(chat_id, st["file_name"])

# ─────────── HELP ───────────
HELP = (
    "Я генерирую/обновляю файлы кода через API и веду версии.\n\n"
    "Основное:\n"
    "• /use <путь/имя> — выбрать файл (напр. /use code.py или /use web/index.html)\n"
    "• Присылай ПРОМПТ — создам/обновлю файл и пришлю готовый.\n"
    "  Если есть предыдущая версия — применю правки к latest автоматически.\n"
    "• Можно прислать ПРОМПТ + код в блоке из трёх обратных кавычек — правки от присланной версии.\n\n"
    "Навигация:\n"
    "• /files — список файлов\n"
    "• /history [file] — показать версии\n"
    "• /get [file] [vNNN|latest] — прислать указанную версию\n"
    "• /rollback vNNN — сделать выбранную версию новой latest (для текущего файла)\n\n"
    "Настройки:\n"
    "• /model <id>   — выбрать модель (по умолчанию gpt-5)\n"
    "• /lang <name>  — подсказка языка (python, javascript, ...)\n"
    "• /state        — показать текущие настройки\n"
    "• /bundle [all] — ZIP: последние версии всех файлов (или все версии)\n"
)

# ─────────── Команды ───────────
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    st = load_state(update.effective_chat.id)
    await update.message.reply_text(
        HELP + f"\n\nТекущие: file={st['file_name']}; session={st['session_id']}; model={st['model']}; lang={st['language']}"
    )

async def cmd_state(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    st = load_state(update.effective_chat.id)
    await update.message.reply_text(f"file={st['file_name']}\nsession={st['session_id']}\nmodel={st['model']}\nlang={st['language']}")

async def cmd_use(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text("Укажи имя/путь файла: /use code.py")
        return
    name = " ".join(ctx.args).strip()
    ext = file_ext(name).lower()
    st = load_state(update.effective_chat.id)
    st["file_name"] = name
    st["session_id"] = make_session_id_from_path(name)
    st["language"] = EXT_LANG.get(ext, st.get("language") or DEFAULT_LANGUAGE)
    save_state(update.effective_chat.id, st)
    await update.message.reply_text(f"OK. file={st['file_name']} (session={st['session_id']}, lang={st['language']})")

async def cmd_file(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await cmd_use(update, ctx)

async def cmd_model(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text("Укажи модель: /model gpt-5")
        return
    st = load_state(update.effective_chat.id)
    st["model"] = " ".join(ctx.args).strip()
    save_state(update.effective_chat.id, st)
    await update.message.reply_text(f"OK. model={st['model']}")

async def cmd_lang(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text("Укажи язык: /lang python")
        return
    st = load_state(update.effective_chat.id)
    st["language"] = " ".join(ctx.args).strip()
    save_state(update.effective_chat.id, st)
    await update.message.reply_text(f"OK. lang={st['language']}")

async def cmd_files(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    rows = list_files_summary(update.effective_chat.id)
    if not rows:
        await update.message.reply_text("Пока нет файлов. Используй /use <file> и пришли промпт.")
        return
    await update.message.reply_text("Файлы:\n" + "\n".join(rows))

async def cmd_history(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if ctx.args:
        name = " ".join(ctx.args).strip()
        sid, ext = touch_manifest_entry(chat_id, name)
    else:
        st = load_state(chat_id)
        sid, ext = touch_manifest_entry(chat_id, st["file_name"])
    vlist = version_paths(chat_id, sid, ext)
    if not vlist:
        await update.message.reply_text("Нет версий.")
        return
    tail = vlist[-20:]
    lines = [f"• {p.name}" for p in tail]
    await update.message.reply_text(f"История {sid} ({len(vlist)} верс.):\n" + "\n".join(lines))

async def cmd_get(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    target_file = None
    target_ver = "latest"
    if ctx.args:
        if len(ctx.args) == 1:
            a = ctx.args[0]
            if a.startswith("v") or a == "latest":
                target_ver = a
            else:
                target_file = a
        else:
            target_file = " ".join(ctx.args[:-1])
            target_ver = ctx.args[-1]

    if target_file:
        sid, ext = touch_manifest_entry(chat_id, target_file)
    else:
        st = load_state(chat_id)
        sid, ext = touch_manifest_entry(chat_id, st["file_name"])

    sess = session_dir(chat_id, sid)
    if target_ver == "latest":
        p = latest_path(chat_id, sid, ext)
    else:
        p = sess / f"{target_ver}{ext}"
    if not p.exists():
        await update.message.reply_text(f"Версия не найдена: {p.name}")
        return

    with p.open("rb") as f:
        await update.message.reply_document(document=InputFile(f, filename=p.name), caption=f"{sid}/{p.name}")

async def cmd_rollback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text("Использование: /rollback vNNN")
        return
    ver = ctx.args[0]
    chat_id = update.effective_chat.id
    st = load_state(chat_id)
    sid, ext = touch_manifest_entry(chat_id, st["file_name"])
    src = session_dir(chat_id, sid) / f"{ver}{ext}"
    if not src.exists():
        await update.message.reply_text(f"Версия не найдена: {src.name}")
        return
    dst = latest_path(chat_id, sid, ext)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    (OUTPUT_ROOT / f"{sid}-latest{ext}").write_text(dst.read_text(encoding="utf-8"), encoding="utf-8")
    await update.message.reply_text(f"OK. latest теперь = {ver}.")

def build_bundle_zip(chat_id: int, latest_only: bool = True) -> Path:
    m = load_manifest(chat_id)
    buf_name = f"bundle-{'latest' if latest_only else 'all'}-{chat_id}.zip"
    out_path = OUTPUT_ROOT / buf_name
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for sid, ent in m["files"].items():
            ext = ent["ext"]
            lp = latest_path(chat_id, sid, ext)
            if lp.exists():
                z.write(lp, arcname=ent["file_name"])
            if not latest_only:
                for v in version_paths(chat_id, sid, ext):
                    z.write(v, arcname=f"{ent['file_name']}.versions/{v.name}")
    return out_path

async def cmd_bundle(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    latest_only = True
    if ctx.args and ctx.args[0].lower() == "all":
        latest_only = False
    p = build_bundle_zip(update.effective_chat.id, latest_only)
    with p.open("rb") as f:
        await update.message.reply_document(
            document=InputFile(f, filename=p.name),
            caption=("Последние версии всех файлов" if latest_only else "Все версии всех файлов")
        )

# ─────────── Основная обработка промптов ───────────
async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text or ""
    prompt, injected_code = parse_msg(text)
    if not prompt:
        await update.message.reply_text("Дай спецификацию (промпт)."); return

    st = load_state(chat_id)
    sid, ext = touch_manifest_entry(chat_id, st["file_name"])

    # база для правок: присланный код или latest
    base_code = injected_code
    if not base_code:
        lp = latest_path(chat_id, sid, ext)
        if lp.exists():
            base_code = lp.read_text(encoding="utf-8")
    mode = "edit" if base_code else "create"

    await ctx.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    try:
        r = requests.post(
            f"{CODEGEN_API_URL}/generate",
            headers={"Content-Type": "application/json"},
            json={
                "prompt": prompt,
                "base_code": base_code,
                "mode": mode,
                "language": st["language"],
                "model": st["model"],
                "session_id": sid,
                "ext": ext
            },
            timeout=180
        )
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code}: {r.text}")
        data = r.json()
        code = data.get("code", "")
        if not code:
            raise RuntimeError("API не вернул поле 'code'")
        diff = data.get("diff")
    except Exception as e:
        await update.message.reply_text(f"Ошибка API: {e}")
        return

    # сохранить новую версию и latest
    vpath = next_version_path(chat_id, sid, ext)
    vpath.write_text(code, encoding="utf-8")
    latest = latest_path(chat_id, sid, ext)
    latest.write_text(code, encoding="utf-8")
    (OUTPUT_ROOT / f"{sid}-latest{ext}").write_text(code, encoding="utf-8")
    ver_num = register_new_version(chat_id, sid, ext)

    # diff, если не прислал API
    if base_code and not diff:
        diff = "\n".join(difflib.unified_diff(
            base_code.splitlines(), code.splitlines(),
            fromfile=f"before{ext}", tofile=f"after{ext}", lineterm=""
        ))

    # отправка файла
    basename = Path(st["file_name"]).name
    with vpath.open("rb") as f:
        await update.message.reply_document(
            document=InputFile(f, filename=basename),
            caption=f"✅ {basename}  (session={sid}, v{ver_num:03d})"
        )

    if diff:
        if len(diff) <= 3500:
            await update.message.reply_text(f"Изменения:\n{diff[:3900]}")
        else:
            dpath = vpath.with_suffix(".diff.txt")
            dpath.write_text(diff, encoding="utf-8")
            with dpath.open("rb") as f:
                await update.message.reply_document(
                    document=InputFile(f, filename=f"{sid}.diff.txt"),
                    caption="Изменения (diff)"
                )

# ─────────── main ───────────
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler("state", cmd_state))
    app.add_handler(CommandHandler("use", cmd_use))
    app.add_handler(CommandHandler("file", cmd_file))  # alias
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("lang", cmd_lang))
    app.add_handler(CommandHandler("files", cmd_files))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(CommandHandler("get", cmd_get))
    app.add_handler(CommandHandler("rollback", cmd_rollback))
    app.add_handler(CommandHandler("bundle", cmd_bundle))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()