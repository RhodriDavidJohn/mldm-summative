
import os
import pandas as pd
import logging
from datetime import datetime
import sys


def setup_logger(run_id: str, folder: str) -> logging.Logger:
    
    logging_file = os.path.join(folder, f"{run_id}.log")
    os.makedirs(os.path.dirname(logging_file), exist_ok=True)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S',
                        filename=logging_file,
                        filemode='w')
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')

    console.setFormatter(formatter)

    logging.getLogger('').addHandler(console)

    LOGGER = logging.getLogger(f'MLDM-{run_id}.logger')

    return LOGGER


def create_run_id(folder: str) -> str:

    date = datetime.now().strftime("%Y-%m-%d") + "_"

    try:
        prev_id = os.listdir(folder)[0]
        prev_date = prev_id[0:11]
        if prev_date == date:
            run_id = int(prev_id[-6:-4]) + 1
        else:
            run_id = 1
    except Exception as e:
        run_id = 1

    run_id = str(run_id).zfill(2)
    run_id = date + run_id

    return run_id


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
