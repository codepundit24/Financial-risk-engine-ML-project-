FROM python:3.11-slim

# Working directory setup
WORKDIR /app

# System dependencies for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/list/*

# Install python depencencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and trained model
COPY src/ ./src/
COPY models/ ./models/

# Expose fastapi port
EXPOSE 8000

# Run Uvicorn
CMD [ "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000" ]