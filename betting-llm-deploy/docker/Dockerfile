# Base image
FROM python:3.11-slim

# Working directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY api/ /app/api/
COPY src/ /app/src/
COPY model/ /app/model/  # LoRA adapter

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

