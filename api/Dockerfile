FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# исходники и стартовый скрипт
COPY . .
RUN chmod +x /app/start.sh

# запуск
CMD ["/app/start.sh"]