import os
from datetime import datetime
from dotenv import load_dotenv

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_base_dir():
    load_dotenv()
    base_dir = os.getenv("BASE_DIR")
    if not base_dir:
        print("WARNING: BASE_DIR not set in environment variables, returning None for base_dir")
    return base_dir

def get_data_dir():
    base_dir = get_base_dir()
    if base_dir:
        data_dir = os.path.join(base_dir, 'data')
    else:
        print("WARNING: BASE_DIR not set in environment variables, returning None for data_dir")
        data_dir = None
    return data_dir