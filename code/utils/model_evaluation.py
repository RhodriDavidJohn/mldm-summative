# imports
import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, auc



def load_model(filepath: str):

    try:
        model = joblib.load(filepath)
        print(f"Model loaded successfully: {filepath}")
        return model
    except Exception as e:
        print(f"Failed to load model {filepath}")
        raise(e)


def get_metrics(model_type: str, model: Pipeline, data_name: str,
                X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Function to get the following evaluation metrics
    1) AUC
    2) F1 score
    """

    # get model predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # calculate metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)

    metrics_df = pd.DataFrame({
        (None, "Data"): [data_name],
        (model_type, "AUC"): [round(auc, 2)],
        (model_type, "F1 score"): [round(f1, 2)]
    })

    return metrics_df


def plot_roc(model_type: str, data_name: str, model: Pipeline,
             X_test: pd.DataFrame, y_test: pd.Series, filepath: str):
    """
    Function to plot the ROC curve
    """

    # Get model predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {model_type} - {data_name}')
    plt.legend(loc='lower right')
    plt.grid()

    # Save the plot
    plt.savefig(filepath)
    plt.close()
    print(f"ROC curve saved to {filepath}")


def plot_shap(model_type: str, model_type_long: str, model: Pipeline,
              X_test: pd.DataFrame, filepath: str):
    """
    Function to plot SHAP feature importance
    """
    
    # preprocess the data
    X_test_transformed = model.named_steps["preprocessing"].transform(X_test)

    # extract the model part of pipeline
    ml_model = model.named_steps[model_type]

    # Check if the model has a predict_proba method
    if not hasattr(ml_model, "predict_proba"):
        print(f"Model {model_type_long} does not support SHAP explanations.")
        return None

    # Initialize SHAP explainer
    if model_type=='lreg':
        explainer = shap.Explainer(ml_model, X_test_transformed)
    elif model_type=='mlp':
        explainer = shap.KernelExplainer(ml_model.predict, X_test_transformed)
    shap_values = explainer(X_test_transformed)

    # Plot SHAP summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_transformed, show=False)
    
    # Save the plot
    plt.savefig(filepath)
    plt.close()
    print(f"SHAP feature importance plot saved to {filepath}")


def model_evaluation(model_type: str, model: Pipeline, data_name: str,
                     X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    
    if model_type=="lreg":
        model_type_long = "LASSO Regression"
    elif model_type=="mlp":
        model_type_long = "MLP"
    else:
        msg = "Model type must be either 'lreg' or 'mlp'."
        raise(KeyError(msg))
    
    data_name_long = data_name.replace('_', ' ')

    metrics = get_metrics(model_type_long, model, data_name_long, X_test, y_test)

    roc_filepath = f"results/roc_plots/{model_type}_{data_name}_model_roc.png"
    plot_roc(model_type_long, data_name_long, model, X_test, y_test, roc_filepath)

    shap_filepath = f"results/shap_plots/{model_type}_{data_name}_model_shap.png"
    plot_shap(model_type, model_type_long, model, X_test, shap_filepath)

    return metrics

