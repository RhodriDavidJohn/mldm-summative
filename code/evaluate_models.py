# imports
import os
import pandas as pd
import joblib
from sys import argv

from code.utils import helpers2 as hlp
from code.utils.model_evaluation import *



def evaluate_models(data_name: str) -> None:

    # load the data
    data_path = f"data/models/{data_name}/test.csv"
    data = hlp.load_csv(data_path)

    X_test = data.drop(columns=["patient_id", "death_2years"])
    y_test = data["death_2years"]

    # load the models
    model_path = "results/models"
    for model in os.listdir(model_path):
        if data_name not in model:
            continue

        if "lreg" in model:
            lreg_model_path = os.path.join(model_path, model)
            lreg_model = load_model(lreg_model_path)
        elif "mlp" in model:
            mlp_model_path = os.path.join(model_path, model)
            mlp_model = load_model(mlp_model_path)
    
    # evaluate models
    lreg_metrics = model_evaluation("lreg", lreg_model, data_name, X_test, y_test)
    mlp_metrics = model_evaluation("mlp", mlp_model, data_name, X_test, y_test)

    metrics_df = lreg_metrics.merge(mlp_metrics, on=(None, "Data"), how="inner")

    save_loc = f"results/{data_name}_model_metrics.csv"
    hlp.save_csv(metrics_df, data_name, save_loc)

    return None


if __name__=="__main__":
    data_name = argv[0]
    evaluate_models(data_name)
