# imports
import os
import pandas as pd
import numpy as np

from utils import helpers2 as hlp

pd.set_option('future.no_silent_downcasting', True)



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


def process_clinical1_data(filepath) -> pd.DataFrame:
    """
    Process first clinical dataset.
    The processing steps are:

    1) clean column names
    2) rename columns
    3) categorise data
    4) derive 2 year survival feature

    Finally, the processed data are saved to the clean data folder.
    """

    df = hlp.load_csv(filepath)

    df = clean_column_names(df)

    rename_cols = {
        'patientid': 'patient_id'
    }
    df = df.rename(columns=rename_cols)

    stage_categories = {
        'I': 1, 'II': 2, 'IIIa': 3, 'IIIb': 3
    }
    df.replace(stage_categories, inplace=True)

    # for ordered categories replace null values with -1
    df['clinical_t_stage'] = df['clinical_t_stage'].fillna(-1)
    df['clinical_n_stage'] = df['clinical_n_stage'].fillna(-1)
    df['clinical_m_stage'] = df['clinical_m_stage'].fillna(-1)
    df['overall_stage'] = df['overall_stage'].fillna(-1)

    df['histology'] = df['histology'].str.replace(' ', '_')

    # N stages go from N0-N3, therefore mark N4 as null
    df['clinical_n_stage'] = df['clinical_n_stage'].replace({4: np.nan})

    # derive 2 year survival
    df['survivaltime_yrs'] = df['survival_time']/364.25
    # remove people who have are still alive but their survival
    # time is less than 2 years
    df = df[~((df['survivaltime_yrs']<2)&(df['deadstatus_event']==0))].reset_index(drop=True).copy()

    df['death_2years'] = [1 if ((df.loc[i, 'survivaltime_yrs']<2)&(df.loc[i, 'deadstatus_event']==1))
                          else 0 for i in range(len(df))]
        

    # drop columns that wouldn't be available in clinical setting
    drop_cols = ['survival_time', 'deadstatus_event', 'survivaltime_yrs']
    df.drop(columns=drop_cols, inplace=True)

    # save cleaned dataset
    output_path = 'data/clean/clinical1_clean.csv'
    hlp.save_csv(df, 'cleaned clinical1 data', output_path)

    return df


def process_clinical2_data(filepath) -> pd.DataFrame:
    """
    Process second clinical dataset.
    The processing steps are:

    1) clean column names
    2) rename columns
    3) categorise data
    4) derive 2 year survival feature

    Finally, the processed data are saved to the clean data folder.
    """

    df = hlp.load_csv(filepath)

    df = clean_column_names(df)

    rename_cols = {
        'case_id': 'patient_id',
        'patient_affiliation': 'affiliation',
        'age_at_histological_diagnosis': 'age',
        'weight_lbs': 'weight',
        'gg': 'gg_percentage',
        'pathological_t_stage': 'clinical_t_stage',
        'pathological_n_stage': 'clinical_n_stage',
        'pathological_m_stage': 'clinical_m_stage',
        'histopathological_grade': 'overall_stage',
        'pleural_invasion_elastic_visceral_or_parietal': 'pleural_invasion'
    }
    df = df.rename(columns=rename_cols)

    # drop columns that wouldn't be available in clinical setting
    drop_cols = ['recurrance', 'recurrence_location',
                 'date_of_recurrence', 'quit_smoking_year']
    df.drop(columns=drop_cols, inplace=True)

    # make the string values lower case
    obj_cols = [col for col in df.columns if df[col].dtype == 'object']

    for col in obj_cols:
        if col=='patient_id':
            continue
        df[col] = df[col].str.lower()
        
    # if someone has never smoked then make their pack years 0
    # so they don't get imputed with 'impossible' values during model training
    df['pack_years'] = [0 if df.loc[i, 'smoking_status']=='nonsmoker'
                        else df.loc[i, 'pack_years'] for i in range(len(df))]
        
    missing_value_replacement = {
        'notassessed': np.nan, 'not recorded in database': np.nan,
        'notcollected': np.nan, 'unknown': np.nan
    }

    binary_value_replacement = {
        'dead': 1, 'alive': 0
    }

    gg_replacement = {
        '0%': 0, '>0 - 25%': 1, '25 - 50%': 2, '50 - 75%': 3, '75 - < 100%': 4, '100%': 5
    }

    t_stage_replacement = {
        'tis': 0, 't1a': 1, 't1b': 1, 't2a':2, 't2b': 2, 't3': 3, 't4': 4
    }

    n_stage_replacement = {
        'n0': 0, 'n1': 1, 'n2': 2, 'n3': 3
    }

    m_stage_replacement = {
        'm0': 0, 'm1a': 1, 'm1b': 2, 'm1c': 3
    }

    hist_grade_replacement = {
        'g1 well differentiated': 1,
        'other, type i: well to moderately differentiated': 1,
        'g2 moderately differentiated': 2,
        'other, type ii: moderately to poorly differentiated': 2,
        'g3 poorly differentiated': 3
    }

    word_replacements = {
        'nsclc nos (not otherwise specified)': 'nos',
        'african-american': 'african_american',
        'hispanic/latino': 'hispanic_latino',
        'native hawaiian/pacific islander': 'native_islander'
    }

    replacements = [missing_value_replacement, binary_value_replacement, gg_replacement,
                    t_stage_replacement, n_stage_replacement, m_stage_replacement,
                    hist_grade_replacement, word_replacements]
    replacement_dict = {}
    for dictionary in replacements:
        for key, value in dictionary.items():
            replacement_dict[key] = value

    df.replace(replacement_dict, inplace=True)

    # for ordered categories replace null values with -1
    df['gg_percentage'] = df['gg_percentage'].fillna(-1)
    df['clinical_t_stage'] = df['clinical_t_stage'].fillna(-1)
    df['clinical_n_stage'] = df['clinical_n_stage'].fillna(-1)
    df['clinical_m_stage'] = df['clinical_m_stage'].fillna(-1)
    df['overall_stage'] = df['overall_stage'].fillna(-1)

    for col in obj_cols:
        if col=='patient_id':
            continue
        df[col] = df[col].str.replace(' ', '_')

    # derive a variable for survival time
    df['date_of_last_known_alive'] = pd.to_datetime(df['date_of_last_known_alive'])
    df['ct_date'] = pd.to_datetime(df['ct_date'])
    df['survival_time_dv'] = ((df['date_of_last_known_alive'] - df['ct_date']).dt.days
                              - df['days_between_ct_and_surgery'])
    df['survival_time_dv'] = [df.loc[i, 'survival_time'] if pd.isna(df.loc[i, 'survival_time_dv'])
                              else df.loc[i, 'survival_time_dv'] for i in df.index]
    df['survivaltime_yrs'] = df['survival_time_dv']/364.25

    # remove people who have are still alive but their survival
    # time is less than 2 years
    df = df[~((df['survivaltime_yrs']<2)&(df['survival_status']==0))].reset_index(drop=True).copy()

    df['death_2years'] = [1 if ((df.loc[i, 'survivaltime_yrs']<2)&(df.loc[i, 'survival_status']==1))
                          else 0 for i in range(len(df))]
        

    # drop columns that wouldn't be available in clinical setting
    drop_cols = ['date_of_last_known_alive', 'survival_status', 'date_of_death',
                 'time_to_death_days', 'ct_date', 'days_between_ct_and_surgery',
                 'pet_date', 'survival_time_dv', 'survivaltime_yrs']
    df.drop(columns=drop_cols, inplace=True)

    # save cleaned dataset
    output_path = 'data/clean/clinical2_clean.csv'
    hlp.save_csv(df, 'cleaned clinical2 data', output_path)

    return df
