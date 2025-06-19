import logging
import os

# Ensure outputs directory exists
os.makedirs('outputs', exist_ok=True)

# Set up logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(os.path.join('outputs', 'case.log'))
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)
