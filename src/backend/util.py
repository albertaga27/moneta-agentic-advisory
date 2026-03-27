"""
Utility functions for the Moneta backend.
"""
import logging
from dotenv import load_dotenv


def load_dotenv_from_azd():
    """Load environment variables from .env file."""
    logging.info("Loading environment from .env file...")
    load_dotenv()
