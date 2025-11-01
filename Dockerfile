# -------------------------------
# NEURA.AI v5 Render Dockerfile
# -------------------------------

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8080

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV PORT=8080

# Run Gunicorn server
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:app"]
