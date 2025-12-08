# Stage 1 omitted (same)

# Stage 2: final
FROM python:3.11-slim

# set WORKDIR to /workspace to match runtime logs
WORKDIR /workspace

# create non-root user (recommended)
RUN useradd --create-home appuser
USER appuser

# copy wheels and install (ensure these paths still exist)
COPY --chown=appuser:appuser --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# copy app code (fix the typo)
COPY --chown=appuser:appuser service/inference_service.py ./inference_service.py
COPY --chown=appuser:appuser service/ ./service/
COPY --chown=appuser:appuser job_runner.py ./job_runner.py

# copy artifacts into /workspace/artifacts (exact path code expects)
COPY --chown=appuser:appuser service/artifacts/ ./artifacts/

ENV PYTHONUNBUFFERED=1
ENV TMPDIR=/tmp

ENTRYPOINT ["python", "job_runner.py"]
