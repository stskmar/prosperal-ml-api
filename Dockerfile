# Stage 1: build
FROM python:3.11-slim AS build

# system deps for common numeric libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy requirements (you may maintain requirements.txt in repo)
COPY service/requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip wheel --no-deps --wheel-dir=/wheels -r requirements.txt

# Stage 2: final
FROM python:3.11-slim

WORKDIR /app

# create non-root user (recommended)
RUN useradd --create-home appuser
USER appuser

# copy wheels and install
COPY --chown=appuser:appuser --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# copy app code
COPY --chown=appuser:appuser service/inference_service.py ./inference_service.p
COPY --chown=appuser:appuser service/ ./service/
COPY --chown=appuser:appuser job_runner.py ./job_runner.py

# working dir
ENV PYTHONUNBUFFERED=1
ENV TMPDIR=/tmp

# entrypoint
ENTRYPOINT ["python", "job_runner.py"]