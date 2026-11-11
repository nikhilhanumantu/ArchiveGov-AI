FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system libraries for OpenCV and document generation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source code and reference data
COPY . .

# Ensure storage directories exist
RUN mkdir -p uploads/originals uploads/working uploads/restored uploads/masks uploads/exports backend/data

# Expose FastAPI application port
EXPOSE 8000

# Seed database and start uvicorn server
CMD ["sh", "-c", "python scripts/seed_database.py && uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"]

