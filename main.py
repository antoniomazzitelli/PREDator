# main.py

import yaml
from src.pipeline import run_pipeline
from src.utils import resolve_path

import sys
import os


def run_main_pipeline(dataset_path, forecast_horizon, look_back, n_epochs, target_var, rnn_flag, output_folder=None):
    """
    Entry point for the pipeline. Loads config, injects the GUI-selected dataset path,
    and delegates execution to run_pipeline.

    Args:
        dataset_path (str): full path to the CSV selected in the GUI
    Returns:
        final_rmse (float): RMSE of the trained model
    """

    # Resolve config path — next to the .app when compiled, cwd in development
    if hasattr(sys, "_MEIPASS"):
        exe_dir = os.path.dirname(sys.executable)
        root = os.path.abspath(os.path.join(exe_dir, "../../.."))
    else:
        root = os.path.abspath(os.getcwd())
    config_path = os.path.join(root, "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Single-dataset assumption — name is fixed as "dataset"
    dataset_name = "dataset"

    # Override the config file path with the one selected at runtime in the GUI
    config[dataset_name]["file"] = dataset_path

    final_rmse = run_pipeline(dataset_name, config, forecast_horizon, look_back, n_epochs, target_var, rnn_flag, output_folder)

    return final_rmse


if __name__ == "__main__":
    import sys
    import os

    def get_config_path():
        """Same path resolution logic as gui.py — kept local to avoid circular imports."""
        if hasattr(sys, "_MEIPASS"):
            exe_dir = os.path.dirname(sys.executable)
            root = os.path.abspath(os.path.join(exe_dir, "../../.."))
        else:
            root = os.path.abspath(os.getcwd())
        return os.path.join(root, "config.yaml")

    with open(get_config_path(), "r") as f:
        cfg = yaml.safe_load(f)

    dataset_path = cfg["dataset"]["file"]
    run_main_pipeline(dataset_path)