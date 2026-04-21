"""
Module: main
Purpose: Application entry point for the machine_learning_model package.
         Configures the loguru logger (stderr + rotating file), sets up
         the logs/ directory, and delegates to application logic.
Rationale: Keeping startup/logging concerns separate from business logic
           makes both independently testable and the entry point small.
Assumptions: The process working directory is the project root so that
             the relative logs/ path resolves correctly.
Failure Modes: Any unhandled exception in main() is caught, logged, and
               exits with code 1 so the process communicates failure to
               the shell or container orchestrator.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

# Configure logger
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{ time: YYYY-MM-DD HH: mm: ss}</green> | <level>{ level: <8}</level> | <cyan>{ name}</cyan>: <cyan>{ function}</cyan>: <cyan>{ line}</cyan> - <level>{ message}</level>",
    level="INFO",
)
logger.add(
    "logs/app.log",
    rotation="10 MB",
    retention="10 days",
    level="DEBUG",
    format="{ time: YYYY-MM-DD HH: mm: ss} | { level: <8} | { name}: { function}: { line} - { message}",
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
        logger.error(f"Application error: { e}")
        sys.exit(1)

    logger.info("Application completed successfully")


if __name__ == "__main__":
    main()
