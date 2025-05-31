# Use a lightweight Python base image
FROM python:3.10-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .

# Install Python dependencies
# Use --no-cache-dir to avoid storing cache in the image
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application code
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI application using uvicorn
# --host 0.0.0.0 makes it accessible from outside the container
# --port 8000 is the port inside the container
# rag_app:app refers to the 'app' object in 'rag_app.py'
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
