# imports
import os
import pandas as pd
import numpy as np
import time
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.utils import resample
import joblib

from utils import helpers as hlp


def pre_processing_pipeline(df: pd.DataFrame, mixed_cols: bool) -> Pipeline:
    """
    Function to create the pre processing pipeline.
    The pre processing steps are:
    1) impute
    2) scale/encode
    """

    if mixed_cols:

        ordinal_cols = [
            'clinical_t_stage',
            'clinical_n_stage',
            'clinical_m_stage',
            'overall_stage'
        ]

        if 'gg_percentage' in df.columns:
            ordinal_cols.append('gg_percentage')

        categorical_cols = [col for col in df.columns if df[col].dtype=='object']
        numerical_cols = [col for col in df.columns
                          if ((df[col].dtype!='object')&(col not in ordinal_cols))]
            
        ordinal_process = Pipeline([('impute', SimpleImputer(strategy='most_frequent'))])
        categorical_process = Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='infrequent_if_exist'))
        ])
        numerical_process = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('encode', MinMaxScaler())
        ])

        pipe = ColumnTransformer(transformers=[
            ('numeric_processing', numerical_process, numerical_cols),
            ('oridnal_processing', ordinal_process, ordinal_cols),
            ('categorical_processing', categorical_process, categorical_cols)
        ])

    else:
        pipe = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', MinMaxScaler())
        ])
        
    return pipe


def model_development(model_type: str, model, params: dict, k_folds: int,
                      data_dict: dict, data_name: str):
    """
    Develop model using pre-processing pipeline
    and grid search cross-validation to tune hyperparameters.
    """

    # logic to choose what columns need pre processing
    if data_name=='image_features':
        mixed_cols = False
    else:
        mixed_cols = True

    pre_processing_pipe = pre_processing_pipeline(data_dict['X_train'], mixed_cols)

    pipe = Pipeline([
        ('preprocessing', pre_processing_pipe),
        (model_type, model)
    ])

    # initiate gridsearchcv object
    pipe_cv = GridSearchCV(
        pipe,
        params,
        scoring='roc_auc',
        refit=True,
        cv=k_folds,
        n_jobs=-1,
        verbose=1
    )

    # fit the gridsearchcv
    try:
        # start timer
        start_time = time.time()

        pipe_cv.fit(data_dict['X_train'], data_dict['y_train'])

        # end the timer
        end_time = time.time()
        hours, remainder = divmod(end_time - start_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        msg = (f"The cross validation process for the {model_type} model "
               f"trained on the {data_name.replace('_', ' ')} data "
               f"took {round(hours, 0)} hrs {round(minutes,0)} mins {round(seconds, 0)} secs.")
        print(msg)
    except Exception as e:
        msg = (f"An error occured while training the {model_type} model "
               f"on the {data_name.replace('_', ' ')} data: {e}")
        print(msg)
        raise(e)
    
    # display best hyperparameters
    best_params = pipe_cv.best_params_
    msg = (f"The best {model_type} model trained on the "
           f"{data_name.replace('_', ' ')} data had the following "
           f"hyperparameters from cross validation: {best_params}")
    print(msg)
        
    # display the AUC score
    auc = pipe_cv.best_score_
    msg = (f"The best {model_type} model trained on the "
           f"{data_name.replace('_', ' ')} data had an AUC "
           f"score of {round(auc, 2)} during cross validation.")
    print(msg)

    return pipe_cv.best_estimator_


def save_ml_model(model, model_name: str, data_name: str,
                  output_path: str) -> None:

    try:
        joblib.dump(model, output_path)
        msg = (f"The {model_name} model trained on the "
               f"{data_name.replace('_', ' ')} data saved successfully to "
               f"{output_path}")
        print(msg)
    except Exception as e:
        msg = (f"Error saving the {model_name} model "
               f"trained on the {data_name.replace('_', ' ')} data to "
               f"{output_path}: {e}")
        print(msg)
        raise(e)
    
    return None


def get_train_test(data: pd.DataFrame, data_name: str, test_size: float, random_state: int) -> tuple:

    X = data.drop(columns=['death_2years']).copy()
    y = data['death_2years'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    train_data = X_train.join(y_train)
    test_data = X_test.join(y_test)

    X_train = X_train.drop("patient_id", axis=1)
    X_test = X_test.drop("patient_id", axis=1)

    train_data_msg = f"training data for {data_name.replace('_', ' ')}"
    test_data_msg = f"testing data for {data_name.replace('_', ' ')}"

    model_data_path = os.path.join(
        'data', 'models', data_name
    )

    hlp.save_csv(train_data, train_data_msg, os.path.join(model_data_path, 'train.csv'))
    hlp.save_csv(test_data, test_data_msg, os.path.join(model_data_path, 'test.csv'))

    return (X_train, X_test, y_train, y_test)


def load_data(input_dir: str, n_batches: int) -> dict:

    data_path_dict = {
        **{'clinical1': os.path.join(input_dir, 'clinical1.csv'),
        'clinical2': os.path.join(input_dir, 'clinical2.csv'),
        'clinical_joined': os.path.join(input_dir, 'clinical_joined.csv')},
        **{f'image_features_{i}': os.path.join(input_dir, f'image_features_{i}.csv') for i in range(1, n_batches+1)}
    }

    image_features_list = []
    for key in data_path_dict.keys():
        if 'image_features' in key:
            image_features_list.append(
                hlp.load_csv(data_path_dict[key])
            )
    image_features = pd.concat(image_features_list, axis=0)

    data_dict = {
        'clinical1': hlp.load_csv(data_path_dict['clinical1']),
        'clinical2': hlp.load_csv(data_path_dict['clinical2']),
        'clinical_joined': hlp.load_csv(data_path_dict['clinical_joined']),
        'image_features': image_features
    }

    data_dict['full_data'] = (
        data_dict['clinical_joined']
        .merge(right=data_dict['image_features'].drop('death_2years', axis=1),
               on='patient_id',
               how='left')
    )

    return data_dict
