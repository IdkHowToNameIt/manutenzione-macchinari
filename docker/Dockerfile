# Immagine del servizio di manutenzione predittiva.
# Il contesto di build e' la radice del progetto (vedi docker/docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MANUTENZIONE_HOST=0.0.0.0 \
    MANUTENZIONE_PORT=8000

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Sorgenti, configurazione e versioni del modello: la versione attiva e' quella
# indicata da models/active_version.json dentro l'immagine.
COPY pyproject.toml ./
COPY src ./src
COPY config ./config
COPY models ./models
RUN pip install --no-cache-dir --no-deps -e . \
    && mkdir -p /app/logs

EXPOSE 8000

CMD ["python", "-m", "manutenzione.serving.app"]
