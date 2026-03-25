# ─────────────────────────────────────────────────────────────────
# Stage: Base image
# Using slim Python image to keep size small
# ─────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# ─────────────────────────────────────────────────────────────────
# Install system dependencies required by Pillow and TensorFlow
# ─────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────────
# Install Python dependencies first (cached layer if unchanged)
# ─────────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────────────────────
# Copy the application files into the container
# (dataset/ and __pycache__/ are excluded via .dockerignore)
# ─────────────────────────────────────────────────────────────────
COPY app.py .
COPY calories.json .
COPY model/ ./model/
COPY templates/ ./templates/
COPY static/ ./static/

# Create the uploads folder so the app can save images
RUN mkdir -p static/uploads

# ─────────────────────────────────────────────────────────────────
# Expose port 5000 (Flask default)
# ─────────────────────────────────────────────────────────────────
EXPOSE 5000

# ─────────────────────────────────────────────────────────────────
# Environment variables
# ─────────────────────────────────────────────────────────────────
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ─────────────────────────────────────────────────────────────────
# Start the Flask app using gunicorn (production-grade WSGI server)
# - 1 worker (enough for single-user/demo use)
# - Timeout 120s to allow model loading on startup
# ─────────────────────────────────────────────────────────────────
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app:app"]
