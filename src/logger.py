import logging ##
import os
from datetime import datetime

LOGS_DIR="logs"
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, F"log_{datetime.now().strftime('%Y-%m-%d')}.log")

logging.basicConfig(
    filename=LOG_FILE, #Wriet all the logs in this file.
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO,     #Remeber there are 5 different levels of logging Debug, Info, Warning, Error, Critical in increasing order. So when you write logging.INFO logs everything at that level and above. This debug is not logged.
)

def get_logger(name):
    logger=logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
    