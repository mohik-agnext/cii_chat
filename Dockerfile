FROM python:3.10-slim

WORKDIR /app

# Install required packages
RUN pip install --no-cache-dir flask flask-cors gunicorn gevent requests

# Copy all necessary files
COPY hybrid_search_frontend.html /app/
COPY simple_server.py /app/

# Set environment variables
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Run the simple server
CMD ["gunicorn", "-k", "gevent", "-w", "1", "--bind", "0.0.0.0:8080", "simple_server:app"]
