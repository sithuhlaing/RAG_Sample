# Dockerfile
FROM python:3.10-slim-bookworm

WORKDIR /app

# Copy the requirements file into the container and install dependencies.
COPY requirements.txt .

# Install Python dependencies
# Use --no-cache-dir to avoid storing cache in the image
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code and data into the container
COPY main.py .
COPY data/ ./data/

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
# For production, you would typically remove --reload
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]