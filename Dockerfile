# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first to leverage Docker's layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
# This is required because your Python script uses parents[1] to find the root
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Run the app. 
# --app-dir tells uvicorn to look inside the 06-inference-api folder for the code
CMD ["uvicorn", "stage6_inference_api:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "06-inference-api"]