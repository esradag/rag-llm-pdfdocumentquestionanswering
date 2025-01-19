# Base image
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port used by Streamlit
EXPOSE 8501

# Add a healthcheck for the Streamlit app
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Define the default command to run the Streamlit app
ENTRYPOINT ["streamlit", "run", "streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]

