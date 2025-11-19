FROM python:3.11-slim

# Evitar que Python genere .pyc y mejorar logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias del sistema (librosa, etc. pueden necesitar)
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalarlos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código
COPY . .

# Cloud Run usa la variable PORT, por defecto 8080
ENV PORT=8080

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
