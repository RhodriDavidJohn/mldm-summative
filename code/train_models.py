# imports
import os
import pandas as pd
from sys import argv

from utils.model_development import *
from utils import helpers as hlp


def train_models(data_name: str, test_size: float, k_folds: int,
                 random_seed: int, n_batches: int) -> None:

    # load the data
    input_dir = 'data/clean'
    data_dict = load_data(input_dir, n_batches)

    try:
        assert data_name in data_dict.keys()
    except Exception as e:
        print("Chosen data name is not valid. Data name must be in",
              f"{list(data_dict.keys())}.")
        raise(e)
    

    data = data_dict[data_name]

    # split the data into train and test set
    X_train, X_test, y_train, y_test = get_train_test(data, data_name, test_size, random_seed)

    model_data = {
        "X_train": X_train,
        "y_train": y_train
    }

    # train the logistic regression model
    lreg_model = LogisticRegression(penalty='l1', solver='liblinear', random_state=random_seed)
    lreg_params = {"lreg__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

    print("Training the logistic regression model on",
           f"{data_name.replace('_', ' ')} data.")

    lreg_model = model_development("lreg", lreg_model, lreg_params, k_folds, model_data, data_name)
    
    lreg_save_loc = f"results/models/lreg_{data_name}_model.pkl"
    save_ml_model(lreg_model, "lreg", data_name, lreg_save_loc)


    # train the MLP model
    mlp_model = MLPClassifier(activation='relu', solver='adam', random_state=random_seed)
    mlp_params = {
        "mlp__hidden_layer_sizes": [(100,), (100,100), (100,50), (100,25), (100,50,50), (100,50,25)],
        "mlp__alpha": [0.001, 0.01, 0.1],
        "mlp__learning_rate": ['constant', 'adaptive', 'invscaling'],
        "mlp__batch_size": [32, 64],
        "mlp__max_iter": [1250, 1500, 1750, 2000, 2250, 2500]
    }

    print("Training the MLP model on",
          f"{data_name.replace('_', ' ')} data.")

    mlp_model = model_development("mlp", mlp_model, mlp_params, k_folds, model_data, data_name)

    mlp_save_loc = f"results/models/mlp_{data_name}_model.pkl"
    save_ml_model(mlp_model, "mlp", data_name, mlp_save_loc)

    return None


if __name__=="__main__":
    data_name = argv[1]
    seed = int(argv[2])
    k_folds = int(argv[3])
    n_batches = int(argv[4])
    test_size = float(argv[5])

    train_models(data_name, test_size, k_folds, seed, n_batches)
