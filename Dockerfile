# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements or install dependencies directly
RUN pip install --no-cache-dir fastapi uvicorn scikit-learn pandas numpy joblib

# Copy the application code and model artifacts into the container
COPY app.py /app/
COPY final_model.pkl /app/
COPY scaler.pkl /app/
COPY imputer.pkl /app/

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]