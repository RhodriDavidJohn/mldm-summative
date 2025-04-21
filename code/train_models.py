# imports
import os
import pandas as pd
from sys import argv

from code.utils.model_development import *
from code.utils import helpers2 as hlp


def train_models(data_name: str, k_folds: int,
                 random_seed: int, n_batches: int) -> None:

    input_dir = 'data/clean'

    data_path_dict = {
        'clinical1': os.path.join(input_dir, 'clinical1.csv'),
        'clinical2': os.path.join(input_dir, 'clinical2.csv'),
        'clinical_joined': os.path.join(input_dir, 'clinical_joined.csv'),
        'image_features_{i}': os.path.join(input_dir, 'image_features_{i}.csv') for i in range(1, n_batches+1)
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
        .merge(right=data_dict['image_features'].drop('death_2years'),
               on='patient_id',
               how='left')
    )


    try:
        assert data_name in data_dict.keys()
    except Exception as e:
        msg = ("Chosen data name is not valid. Data name must be in "
               f"{list(data_dict.keys())}.")
        print(msg)
        raise(e)
    

    data = data_dict[data_name]

    X_train, X_test, y_train, y_test = get_train_test(data, data_name, random_seed)

    model_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

    # Train the logistic regression model
    lreg_model = LogisticRegression(penalty='l1', solver='liblinear',
                                    random_state=random_seed)
    lreg_params = {
        "lreg__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    }

    msg = ("Training the logistic regression model on "
           f"{data_name.replace('_', ' ')} data.")
    print(msg)

    lreg_model = model_development(
            "lreg", lreg_model, lreg_params, k_folds,
            model_data, data_name
        )
    
    lreg_save_loc = f"results/models/lreg_{data_name}_model.pk"
    save_ml_model(lreg_model, "lreg", data_name, lreg_save_loc)


    # train the MLP model
    mlp_model = MLPClassifier(activation='relu', solver='adam',
                              random_state=self.random_seed)
    mlp_params = {
        "mlp__hidden_layer_sizes": [(100,), (100,50), (100,50,50), (100,50,25)],
        "mlp__alpha": [0.001, 0.01, 0.1],
        "mlp__learning_rate": ['constant', 'adaptive', 'invscaling'],
        "mlp__batch_size": [32, 64],
        "mlp__max_iter": [200, 500, 1000]
    }

    msg = ("Training the MLP model on "
                   f"{data_name.replace('_', ' ')} data.")
    print(msg)

    mlp_model = model_development(
        "mlp", mlp_model, mlp_params, k_folds,
        model_data, data_name
    )

    mlp_save_loc = f"results/models/mlp_{data_name}_model.pk"
    save_ml_model(mlp_model, "mlp", data_name, mlp_save_loc)

    return None


if __name__=="__main__":
    data_name = argv[0]
    seed = argv[1]
    k_folds = argv[2]
    n_batches = argv[3]

    train_models(data_name, k_folds, seed, n_batches)
