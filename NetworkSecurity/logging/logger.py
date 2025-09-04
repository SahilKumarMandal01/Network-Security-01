import logging
import os
from datetime import datetime

# Define log directory
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True) # Ensure directory exists

# Define log fine name with timestamp
LOG_FILE = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(levelname)s - %(name)s - line %(lineno)d - %(message)s",
    level=logging.INFO
)

# Example usage
if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("Logging system is set up successfully!")
