
import os
import pandas as pd
import numpy as np
import joblib
import pydicom
from skimage import io
import logging
from datetime import datetime


def setup_logger(folder: str, run_id: str) -> logging.Logger:
    
    logging_file = os.path.join(folder, f"{run_id}.log")
    
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
        prev_id = sorted(os.listdir(folder))[-1]
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
        raise(e)
    
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
        raise(e)
    
    return None


def load_dicom(filepath: str, LOGGER: logging.Logger) -> np.ndarray:

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")

    try:
        dicom_file = pydicom.dcmread(filepath)
        pixel_array = dicom_file.pixel_array
        image = (pixel_array / np.max(pixel_array) * 255).astype(np.uint8)
    except Exception as e:
        LOGGER.error(f"Error reading {filepath}: {e}")
        raise(e)
    
    return image


def save_medical_image(image: np.ndarray,
                       data_name: str,
                       filepath: str,
                       LOGGER: logging.Logger) -> None:
    
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="skimage")

    try:
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        io.imsave(filepath, image)
    except Exception as e:
        LOGGER.error(f"Error saving {data_name} to {filepath}: {e}")
        raise(e)
    
    return None


def calculate_solidarity(region):
    if region.convex_area > 0:
        solidity = region.area / region.convex_area
    else:
        solidity = 0  
    return solidity


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to clean column names.

    The function removes whitespace and special characters, 
    converts to lowercase, and makes the names snake_case.
    """

    df = df.copy()

    # remove whitespace
    df.columns = df.columns.str.strip()

    # remove special characters
    df.columns = df.columns.str.replace('.', ' ')
    df.columns = df.columns.str.replace('[%(),-]', '', regex=True)
    df.columns = df.columns.str.replace('choice=', '')

    # converts to lower case
    df.columns = df.columns.str.lower()

    # make the names snake case
    df.columns = df.columns.str.replace(' ', '_')

    return df


def load_slices(images_folder: str) -> dict:
    """
    Load slices from a folder and return them as a list of numpy arrays within a dictionairy.
    """

    patient_ids = sorted(os.listdir(images_folder))
    slices_dict = {}
    for patient in patient_ids:
        slice_folder = os.path.join(images_folder, patient, 'ct')
        slice_files = sorted(os.listdir(slice_folder))
        slices = [io.imread(os.path.join(slice_folder, file)) for file in slice_files]
        slices_dict[patient] = slices

    return slices_dict


def create_3d_image(slices: list) -> np.ndarray:
    """
    Stack slices to create a 3D image.
    """
    return np.stack(slices, axis=0)


def calculate_glcm2(img, mask, nbins):
    out = np.zeros((nbins,nbins,13))
    offsets = [(1, 0, 0),
                       (0, 1, 0),
                       (0, 0, 1),
                       (1, 1, 0),
                       (-1, 1, 0),
                       (1, 0, 1),
                       (-1, 0, 1),
                       (0, 1, 1),
                       (0, -1, 1),
                       (1, 1, 1),
                       (-1, 1, 1),
                       (1, -1, 1),
                       (1, 1, -1)
                       ]
    matrix = np.array(img)
    matrix[mask <= 0] = nbins
    s= matrix.shape

    bins = np.arange(0,nbins+1)

    for i,offset in  enumerate(offsets):

        matrix1 = np.ravel(matrix[max(offset[0],0):s[0]+min(offset[0],0),max(offset[1],0):s[1]+min(offset[1],0),
                  max(offset[2],0):s[2]+min(offset[2],0)])

        matrix2 = np.ravel(matrix[max(-offset[0], 0):s[0]+min(-offset[0], 0), max(-offset[1], 0):s[1]+min(-offset[1], 0),
                  max(-offset[2], 0):s[2]+min(-offset[2], 0)])


        out[:,:,i] = np.histogram2d(matrix1,matrix2,bins=bins)[0]
    return out


def save_ml_model(model, model_name: str, data_name: str,
                  output_path: str, LOGGER: logging.Logger) -> None:

    save_loc = os.path.join(output_path, f"{data_name}_{model_name}_model.pkl")

    try:
        joblib.dump(model, save_loc)
        msg = (f"The {model_name} model trained on the "
               f"{data_name.replace('_', ' ')} data saved successfully to "
               f"{save_loc}")
        LOGGER.info(msg)
    except Exception as e:
        msg = (f"Error saving the {model_name} model "
               f"trained on the {data_name.replace('_', ' ')} data to"
               f"{save_loc}: {e}")
        LOGGER.error(msg)
        raise(e)
    
    return None


def get_metadata() -> pd.DataFrame:

    meta1 = load_csv("data/metadata1.csv")
    meta2 = load_csv("data/metadata2.csv")

    meta1['Dataset'] = "dataset1"
    meta2['Dataset'] = "dataset2"

    metadata = pd.concat([meta1, meta2], axis=0)

    return metadata
