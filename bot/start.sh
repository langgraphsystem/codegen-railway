#!/usr/bin/env bash
set -euo pipefail

# Собираем .env из переменных окружения (Railway Variables)
: > .env
[ -n "${TELEGRAM_TOKEN:-}" ]    && echo "TELEGRAM_TOKEN=${TELEGRAM_TOKEN}" >> .env
[ -n "${CODEGEN_API_URL:-}" ]   && echo "CODEGEN_API_URL=${CODEGEN_API_URL}" >> .env
echo "DEFAULT_MODEL=${DEFAULT_MODEL:-gpt-5}"        >> .env
echo "DEFAULT_LANGUAGE=${DEFAULT_LANGUAGE:-python}"  >> .env
echo "DEFAULT_FILE_NAME=${DEFAULT_FILE_NAME:-code.py}" >> .env
echo "SESSIONS_DIR=${SESSIONS_DIR:-/data/sessions}"  >> .env
echo "OUTPUT_DIR=${OUTPUT_DIR:-/data/code}"          >> .env

# Папки для версий
mkdir -p /data/sessions /data/code

# Старт бота
exec python tg_bot_api.py