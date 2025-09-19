FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 

WORKDIR /app

# Copy the current directory contents into the container
COPY . /app 

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt


# Expose port 8000 (the port our FastAPI app will run on)
EXPOSE 8000

# Specify the entrypoint to run the app with Uvicorn (listening on all interfaces at port 8000)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]