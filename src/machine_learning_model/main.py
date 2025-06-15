"""
Main module for machine_learning_model
"""

import sys
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logger
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/app.log",
    rotation="10 MB",
    retention="10 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)


def setup_logging():
    """Setup logging configuration."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    logger.info("Logging setup complete")


def main():
    """Main entry point for the application."""
    setup_logging()
    logger.info("Starting machine_learning_model application")

    try:
        logger.info("Hello from machine_learning_model!")
        # Your application logic here

    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

    logger.info("Application completed successfully")


if __name__ == "__main__":
    main()
