"""
ASGI entry point for production deployment with Uvicorn / Gunicorn
Author: Manjunath
"""

from app import app

# Export ASGI application
application = app
