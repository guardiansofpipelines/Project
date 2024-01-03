# Use a base Python Image
FROM python:3.10.13-slim-bookworm

# Set the working directory to /app
WORKDIR /app

# Copy the requirements.txt file into the working directory

COPY requirements.txt .

# Install any needed packages specified in requirements.txt

RUN pip install -r requirements.txt

# Copy directory
COPY . . 

# Expose the flask app port

EXPOSE 5001

# Define the command to run the flask app when the container starts

CMD dvc pull --force && python /app/app/web_app.py