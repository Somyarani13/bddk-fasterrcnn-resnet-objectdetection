# Use the official Python image from DockerHub
FROM python:3.9-slim

# Install system dependencies for OpenCV, including libGL
# The error is related to a hash sum mismatch, which can happen due to network issues or package repository inconsistency.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt into the working directory
COPY ./requirements.txt /app/requirements.txt

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the Streamlit app (app.py) into the container
COPY ./app.py /app/app.py

# port on which the Streamlit app will run
EXPOSE 8501

# Set the entrypoint command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]