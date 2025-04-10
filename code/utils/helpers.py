
import os
import pandas as pd
import logging
import sys


def setup_logger(folder: str) -> logging.Logger:
    pass


def create_run_id(folder: str) -> str:
    pass


def load_csv(filepath: str, LOGGER: logging.Logger) -> pd.DataFrame:

    try:
        df = pd.read_csv(filepath)
        LOGGER.info(f"Loaded {filepath} successfully")
    except Exception as e:
        LOGGER.error(f"Error reading {filepath}: {e}")
        sys.exit()
    
    return df


def save_csv(
        df: pd.DataFrame,
        data_name: str,
        output_path: str,
        LOGGER: logging.Logger
    ) -> None:

    try:
        df.to_csv(output_path, index=False)
        LOGGER.info(f'Saved {data_name} to {output_path}')
    except Exception as e:
        logging.error(f"Error saving {data_name} to {output_path}: {e}")
        sys.exit()