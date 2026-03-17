FROM python:3.11-slim

# System dependencies for python packages (like psycopg2, ccxt)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement list
COPY requirements.txt .

# Upgrade pip and install standard packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy specific source logic
COPY env/ ./env/
COPY agent/ ./agent/
COPY data/ ./data/
COPY api/ ./api/
COPY frontend/ ./frontend/
COPY rag/ ./rag/
COPY models/ ./models/

EXPOSE 8000

# Server execution handled by Docker Compose
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
