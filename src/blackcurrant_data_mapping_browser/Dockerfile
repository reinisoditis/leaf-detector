# Use the official Python 3.10 base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose the FastAPI default port (adjust if necessary)
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "blackcurrant_fast_api:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
