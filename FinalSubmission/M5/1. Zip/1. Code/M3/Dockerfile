# Using the same python runtime as my conda environment
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the contents of the current directory contents to container
COPY . /app

# Install Pip and Python Depemdencies
RUN pip install --no-cache-dir pip --upgrade
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available
EXPOSE 5000

# Define environment variable, Just to track.
ENV NAME=MLOpsAssignment1

# Run m3_flask.py Flask Application when the container launches
CMD ["python", "m3_flask.py"]