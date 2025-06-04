FROM python:3.10-slim

WORKDIR /app

# Install Flask
RUN pip install --no-cache-dir flask flask-cors

# Copy only essential files
COPY hybrid_search_frontend.html /app/
COPY run_server.py /app/

# Set environment variables
ENV PORT=3001

# Expose the port
EXPOSE 3001

# Run the minimal server
CMD ["python", "run_server.py"] 