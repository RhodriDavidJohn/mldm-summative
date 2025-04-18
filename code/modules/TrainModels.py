# imports
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from configparser import ConfigParser
import logging
import time
import warnings

from code.utils import helpers as hlp



class TrainModels:
    def __init__(self, config: ConfigParser, logger: logging.Logger,
                 clinical1: pd.DataFrame, clinical2: pd.DataFrame,
                 clinical_joined: pd.DataFrame, image_features: pd.DataFrame,
                 full_data: pd.DataFrame):
        
        self.output_path = config['output_path']
        self.k_folds = config['k_folds']
        self.random_seed = config['random_seed']
        self.LOGGER = logger
        self.clinical1 = clinical1
        self.clinical2 = clinical2
        self.clinical_joined = clinical_joined
        self.image_features = image_features
        self.full_data = full_data

    
    def train_models(self) -> None:
        """
        Train a logistic regression model and a MLP model
        on each of the datasets.
        """

        data_dict = {'clinical1': self.clinical1.copy(),
                     'clinical2': self.clinical2.copy(),
                     'clinical_joined': self.clinical_joined.copy(),
                     'image_feaures': self.image_features.copy(),
                     'full_data': self.full_data.copy()}
        
        model_data = {}
        for data_name, data in data_dict.items():
            X = data.drop(columns=['patient_id', 'death_2years']).copy()
            y = data['death_2years'].copy()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=self.random_seed
            )

            model_data[data_name] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
        
        self.model_data_dict = model_data


        # save the training and testing data for each dataset
        model_data_path = os.path.join(self.output_path, 'data')
        for data_name, data in model_data.items():
            train_data = data['X_train'].join(data['y_train'])
            test_data = data['X_test'].join(data['y_test'])

            train_data_msg = f"taining data for {data_name.replace('_', ' ')}"
            test_data_msg = f"testing data for {data_name.replace('_', ' ')}"

            hlp.save_csv(train_data, train_data_msg, model_data_path, self.LOGGER)
            hlp.save_csv(test_data, test_data_msg, model_data_path, self.LOGGER)



        # Train the logistic regression models
        lreg_model = LogisticRegression(penalty='l1', solver='liblinear',
                                        random_state=self.random_seed)
        lreg_params = {
            "lreg__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        }

        lreg_models = []
        for data_name, data in data_dict.items():
            msg = ("Training the logistic regression model on "
                   f"{data_name.replace('_', ' ')} data.")
            self.LOGGER.info(msg)

            model = self.model_development(
                "lreg", lreg_model, lreg_params, data, data_name, self.output_path
            )
            lreg_models.append(model)
        
        self.LOGGER.info(f"Trained all logistic regression models and saved to {self.output_path}.")

        self.clinical1_lreg = lreg_models[0]
        self.clinical2_lreg = lreg_models[1]
        self.clinical_lreg = lreg_models[2]
        self.image_lreg = lreg_models[3]
        self.full_lreg = lreg_models[4]


        # Train the mlp models
        mlp_model = MLPClassifier(activation='relu', solver='adam',
                                  random_state=self.random_seed)
        mlp_params = {
            "mlp__hidden_layer_sizes": [(100,), (100,50), (100,50,50), (100,50,25)],
            "mlp__alpha": [0.001, 0.01, 0.1],
            "mlp__learning_rate": ['constant', 'adaptive', 'invscaling'],
            "mlp__batch_size": [32, 64],
            "mlp__max_iter": [200, 500, 1000]
        }

        mlp_models = []
        for data_name, data in data_dict.items():
            msg = ("Training the MLP model on "
                   f"{data_name.replace('_', ' ')} data.")
            self.LOGGER.info(msg)

            model = self.model_development(
                "mlp", mlp_model, mlp_params, data, data_name, self.output_path
            )
            mlp_models.append(model)
        
        self.LOGGER.info(f"Trained all MLP models and saved to {self.output_path}.")

        self.clinical1_mlp = mlp_models[0]
        self.clinical2_mlp = mlp_models[1]
        self.clinical_mlp = mlp_models[2]
        self.image_mlp = mlp_models[3]
        self.full_mlp = mlp_models[4]

        return None
    

    def model_development(self, model_type: str, model, params: dict,
                          data_dict: dict, data_name: str, output_path: str):
        """
        Develop model using pre-processing pipeline
        and grid search cross-validation to tune hyperparameters.
        """

        # logic to choose what columns need pre processing
        if data_name=='image_feaures':
            mixed_cols = False
        else:
            mixed_cols = True

        pre_processing_pipe = self.pre_processing_pipeline(data_dict['X_train'], mixed_cols)

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
            cv=self.k_folds,
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
                   f"took {hours} hrs {minutes} mins {seconds} secs.")
            self.LOGGER.info(msg)
        except Exception as e:
            msg = (f"An error occured while training the {model_type} model "
                   f"on the {data_name.replace('_', ' ')} data: {e}")
            self.LOGGER.error(msg)
            raise(e)
        
        # display the AUC score
        auc = pipe_cv.best_score_
        msg = (f"The best {model_type} model trained on the "
               f"{data_name.replace('_', ' ')} data had an AUC "
               f"score of {round(auc, 2)} during cross validation.")
        self.LOGGER.info(msg)

        # save the model
        best_model = pipe_cv.best_estimator_
        hlp.save_ml_model(best_model, model_type, data_name, output_path, self.LOGGER)

        return best_model


    def pre_processing_pipeline(self, df:pd.DataFrame, mixed_cols: bool) -> Pipeline:
        """
        Function to create the pre processing pipeline.
        The pre processing steps are:
        1) impute
        2) scale/encode
        """

        if mixed_cols:

            ordinal_cols = [
                'gg_percentage',
                'clinical_t_stage',
                'clinical_n_stage',
                'clinical_m_stage',
                'overall_stage'
            ]

            categorical_cols = [col for col in df.columns if df[col].dtype=='object']
            numerical_cols = [col for col in df.columns
                              if ((df[col].dtype!='object')&(col not in ordinal_cols))]
            
            ordinal_process = Pipeline([('impute', SimpleImputer(strategy='most_frequent'))])
            categorical_process = Pipeline([
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('encode', OneHotEncoder())
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
        