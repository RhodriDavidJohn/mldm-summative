
import os
import pandas as pd
import numpy as np
import joblib
import pydicom
from skimage import io
import logging
from datetime import datetime


def load_csv(filepath: str) -> pd.DataFrame:

    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {filepath} successfully")
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        raise(e)
    
    return df


def save_csv(
        df: pd.DataFrame,
        data_name: str,
        output_path: str
    ) -> None:

    try:
        df.to_csv(output_path, index=False)
        print(f'Saved {data_name} to {output_path}')
    except Exception as e:
        print(f"Error saving {data_name} to {output_path}: {e}")
        raise(e)
    
    return None


def get_metadata() -> pd.DataFrame:

    meta1 = load_csv("data/metadata1.csv")
    meta2 = load_csv("data/metadata2.csv")

    meta1['Dataset'] = "dataset1"
    meta2['Dataset'] = "dataset2"

    metadata = pd.concat([meta1, meta2], axis=0)

    return metadata
