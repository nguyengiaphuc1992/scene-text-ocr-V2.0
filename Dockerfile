# Use a base image with TensorFlow and Python
#FROM python:3.8-slim-buster
FROM paddlepaddle/paddle:2.5.1
# Set the working directory
WORKDIR /app

# Install libgomp
# RUN apt-get update && \
#     apt-get install -y libgomp1 && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Copy the FastAPI app code to the container
COPY . .

# Expose the FastAPI port
EXPOSE 80

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]