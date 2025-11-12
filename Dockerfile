# 1. Start with an official Python base image
FROM python:3.10-slim

# 2. Set the "working directory" inside the container
WORKDIR /app

# 3. Copy the requirements file in first (for layer caching)
COPY requirements.txt .

# 4. Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code into the container
COPY . .

# 6. Expose the port Streamlit runs on (default is 8501)
EXPOSE 8501

# 7. Define the command to run when the container starts
# We use "healthcheck" to make sure Streamlit is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# This command runs your app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]