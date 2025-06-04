#!/usr/bin/env python3
import os
from flask import Flask, send_from_directory
from flask_cors import CORS

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def serve_frontend():
    """Serve the frontend HTML file."""
    return send_from_directory('.', 'hybrid_search_frontend.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3001))
    print(f"Starting minimal server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port)