# src/utils.py

import os
import logging
import sys

# ===============================
# Path resolution
# ===============================

def get_project_root():
    """Returns the project root — parent of PREDator.app when compiled, cwd in development."""
    if hasattr(sys, "_MEIPASS"):
        exe_dir = os.path.dirname(sys.executable)
        return os.path.abspath(os.path.join(exe_dir, "../../.."))
    return os.path.abspath(os.getcwd())

def resolve_path(relative_path):
    """Converts a relative path to an absolute path anchored at the project root."""
    root = get_project_root()
    return os.path.join(root, relative_path)

# ===============================
# Directory management
# ===============================

def ensure_dir(path):
    """Creates the directory at path if it does not already exist."""
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Impossibile creare la cartella {path}: {e}")

# ===============================
# Logging
# ===============================

def setup_logging(name, log_dir=None):
    """
    Configures a named logger. Writes to console always, and to file if log_dir is provided.
    Clears existing handlers to avoid duplicate entries on repeated calls.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if log_dir:
        ensure_dir(log_dir)
        log_file = os.path.join(log_dir, f"{name}.log")
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

# ===============================
# Sequence generation
# ===============================

import numpy as np

def create_sequences(data, look_back, forecast_horizon):
    """
    Converts a 1D scaled time series into supervised learning sequences.
    Each sample: input window of length look_back → output of length forecast_horizon.
    """
    X, y = [], []
    for i in range(len(data) - look_back - forecast_horizon + 1):
        X.append(data[i:i+look_back])                        # input window
        y.append(data[i+look_back:i+look_back+forecast_horizon].flatten())  # flattened output
    X = np.array(X)
    y = np.array(y)
    return X, y
