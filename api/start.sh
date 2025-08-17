#!/usr/bin/env bash
set -euo pipefail

# Собираем .env из переменных окружения (Railway Variables)
: > .env
[ -n "${OPENAI_API_KEY:-}" ]  && echo "OPENAI_API_KEY=${OPENAI_API_KEY}" >> .env
[ -n "${OPENAI_BASE_URL:-}" ] && echo "OPENAI_BASE_URL=${OPENAI_BASE_URL}" >> .env
echo "DEFAULT_MODEL=${DEFAULT_MODEL:-gpt-5}"        >> .env
echo "SESSIONS_DIR=${SESSIONS_DIR:-/data/sessions}" >> .env
echo "OUTPUT_DIR=${OUTPUT_DIR:-/data/code}"         >> .env

# Папки для хранения версий
mkdir -p /data /data/sessions /data/code

# Старт API на порту, который задаёт Railway
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"