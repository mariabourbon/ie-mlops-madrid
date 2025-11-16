FROM python:3.11-slim

# System lib needed by scikit-learn/OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Keep pip modern so it can find wheels
RUN python -m pip install --upgrade pip setuptools wheel

# Install runtime deps using wheels only (no source builds)
COPY requirements-api.txt .
RUN pip install --only-binary=:all: --no-cache-dir -r requirements-api.txt

# Copy app & model after deps for better caching
COPY app/ app/
COPY models/ models/

EXPOSE 8000
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]

