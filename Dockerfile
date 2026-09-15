# 1. Use Python 3.12 to match your development environment
FROM python:3.12-slim

# 2. Set the working directory
WORKDIR /app

# 3. Install system dependencies (optional but recommended for some pandas/sklearn builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy and install requirements first to optimize Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the entire project
# Your .dockerignore already handles .venv and .git, so this is efficient
COPY . .

# 6. Expose the FastAPI port
EXPOSE 8000

# 7. Run the API
# Running as a module (06-inference-api.stage6_inference_api:app)
# from the root directory is cleaner and preserves your Path(__file__).resolve().parents[1] logic.
CMD ["uvicorn", "06-inference-api.stage6_inference_api:app", "--host", "0.0.0.0", "--port", "8000"]