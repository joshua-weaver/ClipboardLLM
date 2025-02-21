import logging
import os
import sys
from datetime import datetime

def setup_logger():
    # Set up log directory in AppData for Windows
    if getattr(sys, 'frozen', False):
        # If we're running as a PyInstaller bundle
        app_data = os.getenv('APPDATA')
        base_dir = os.path.join(app_data, 'ClipboardLLM')
    else:
        # If we're running from source
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    logs_dir = os.path.join(base_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a new log file for each session
    log_file = os.path.join(logs_dir, f'clipboard_llm_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create logger
    logger = logging.getLogger('ClipboardLLM')
    logger.info(f"Log file created at: {log_file}")
    logger.info(f"Running from: {os.getcwd()}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Logs directory: {logs_dir}")
    
    return logger