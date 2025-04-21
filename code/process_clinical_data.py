# imports
import os
import pandas as pd

from code.utils.clinical_processing import *
from code.utils import helpers2 as hlp



def process_clinical_data() -> None:

    base_dir = '/user/home/ms13525/scratch/mshds-ml-data-2025'
    clinical1_path = os.path.join(base_dir, 'dataset1', 'clinical1.csv')
    clinical2_path = os.path.join(base_dir, 'dataset2', 'clinical2.csv')

    cleaned_clinical1 = process_clinical_data(clinical1_path)
    cleaned_clinical2 = process_clinical2_data(clinical2_path)

    clinical_joined = pd.concat([cleaned_clinical1, cleaned_clinical2], axis=0)

    # mark null categorical values as unknown or not_applicable
    # for variables that are in one of the datasets but not the other
    cols_to_mark_unkown = ['affiliation', 'smoking_status', 'ethnicity']
    df[cols_to_mark_unkown] = df[cols_to_mark_unkown].fillna('unknown')

    df['pack_years'] = df['pack_years'].fillna(-1)

    # save the joined dataset
    output_path = os.path.join(self.output_path, 'clinical_joined.csv')
    hlp.save_csv(df, 'joined clinical data', output_path, self.LOGGER)

    return None


if __name__=="__main__":
    process_clinical_data()
