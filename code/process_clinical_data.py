# imports
import os
import pandas as pd

from utils.clinical_processing import *
from utils import helpers as hlp



def process_clinical_data() -> None:

    base_dir = '/user/home/ms13525/scratch/mshds-ml-data-2025'
    clinical1_path = os.path.join(base_dir, 'dataset1', 'clinical1.csv')
    clinical2_path = os.path.join(base_dir, 'dataset2', 'clinical2.csv')

    cleaned_clinical1 = process_clinical1_data(clinical1_path)
    cleaned_clinical2 = process_clinical2_data(clinical2_path)

    cols = cleaned_clinical1.columns.tolist()

    clinical_joined = pd.concat([cleaned_clinical1, cleaned_clinical2[cols]], axis=0)

    # save the joined dataset
    output_path = 'data/clean/clinical_joined.csv'
    hlp.save_csv(clinical_joined, 'joined clinical data', output_path)

    return None


if __name__=="__main__":
    process_clinical_data()
