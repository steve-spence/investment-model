# Base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy contents of models folder into /app
COPY models/ .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
