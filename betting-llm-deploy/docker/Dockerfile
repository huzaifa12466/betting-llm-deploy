# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model folder first
COPY model/ /app/model/

# Copy rest of the source code
COPY src/ /app/src/
COPY api/ /app/api/
COPY web/ /app/web/

# Expose port (optional)
EXPOSE 8000

# Set default command to run FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

