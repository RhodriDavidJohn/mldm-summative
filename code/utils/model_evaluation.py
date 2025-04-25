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

    data_name = data_name.title()

    # get model predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # calculate metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)

    metrics_df = pd.DataFrame({
        ("", "Data"): [data_name],
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


def plot_coefs(model: Pipeline, data_name: str, filepath: str):
    """
    Function to plot model coefficients
    """

    # get feature names
    features = model.named_steps["preprocessing"].get_feature_names_out()

    # extract the model part of pipeline
    ml_model = model.named_steps["lreg"]

    # create plot data
    df = pd.DataFrame(ml_model.coef_.T, index=features, columns=['coefficients'])
    df['abs_coefs'] = df['coefficients'].abs()
    df = df.sort_values(by='abs_coefs', ascending=False)
    plot_data = df.head(10).copy()

    # plot the top 10 coefficients
    plt.figure(figsize=(24,14))
    plt.barh(plot_data.index, plot_data.coefficients)

    plt.xlabel("Model Coefficient Value", fontsize=14)
    plt.ylabel("Feature Name", fontsize=14)

    title = (f"Top {len(plot_data)} LASSO model "
             f"coefficients for {data_name} data")
    plt.title(title,
              fontsize=16, pad=20)

    # save the plot
    plt.savefig(filepath)
    plt.close()

    print(f"LASSO coeffincients plot saved to {filepath}")


def plot_shap(model: Pipeline, data_name: str, X_test: pd.DataFrame, filepath: str):
    """
    Function to plot SHAP feature importance
    """
    
    # preprocess the data
    X_test_transformed = model.named_steps["preprocessing"].transform(X_test)

    # get feature names
    features = model.named_steps["preprocessing"].get_feature_names_out()
    try:
        features = [feature.split('__')[1] for feature in features]
    except:
        pass

    X_test_transformed = pd.DataFrame(
        data=X_test_transformed, columns=features
    )

    # extract the model part of pipeline
    ml_model = model.named_steps["mlp"]

    # check if the model has a predict_proba method
    if not hasattr(ml_model, "predict_proba"):
        print(f"MLP model does not support SHAP explanations.")
        return None

    # initialize SHAP explainer
    explainer = shap.KernelExplainer(ml_model.predict, X_test_transformed)
    shap_values = explainer(X_test_transformed)

    # plot SHAP summary plot
    title = (f"SHAP plot (top 10 features): MLP model "
             f"trained on {data_name} data")
    plt.figure()
    shap.summary_plot(shap_values, X_test_transformed,
                      max_display=10, show=False, plot_size=(12,8))
    plt.title(title, fontsize=16, pad=20)
    
    # Save the plot
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f"SHAP feature importance plot saved to {filepath}")


def model_evaluation(model_type: str, model: Pipeline, data_name: str,
                     X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    
    if model_type=="lreg":
        model_type_long = "LASSO"
    elif model_type=="mlp":
        model_type_long = "MLP"
    else:
        msg = "Model type must be either 'lreg' or 'mlp'."
        raise(KeyError(msg))
    
    data_name_long = data_name.replace('_', ' ')

    # auc and f1 metrics
    metrics = get_metrics(model_type_long, model, data_name_long, X_test, y_test)

    # roc plot
    roc_filepath = f"results/roc_plots/{model_type}_{data_name}_model_roc.png"
    plot_roc(model_type_long, data_name_long, model, X_test, y_test, roc_filepath)

    # feature importance plots
    base_path = "results/feature_plots"
    if model_type=="lreg":
        coef_filepath = os.path.join(base_path, f"lreg_{data_name}_model_coefs.png")
        plot_coefs(model, data_name_long, coef_filepath)
    else:
        shap_filepath = os.path.join(base_path, f"mlp_{data_name}_model_shap.png")
        plot_shap(model, data_name_long, X_test, shap_filepath)

    return metrics

