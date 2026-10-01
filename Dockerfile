FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

COPY requirements-cpu.txt requirements-service.txt ./
RUN python -m pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements-cpu.txt

# Include only runtime code; datasets, checkpoints and notebook outputs stay out.
COPY postcode_ml/ ./postcode_ml/
COPY serve.py predict_bundle.py make_submission.py ./
COPY scripts/ ./scripts/
RUN mkdir -p /app/runtime

EXPOSE 8000
VOLUME ["/app/runtime"]
CMD ["python", "serve.py", "--host", "0.0.0.0", "--port", "8000", "--bundle-dir", "runtime/attention518", "--db-path", "runtime/assessments.sqlite3", "--threads", "4"]
