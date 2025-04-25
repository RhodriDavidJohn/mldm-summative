# -*- mode: snakemake -*-
import pandas as pd
from configparser import ConfigParser

from code.utils.helpers import get_metadata

configfile: "config/config.yml"

# get a list of all the patient IDs that have
# segmented images
metadata = get_metadata(surpress_messages=True)
seg_ids = metadata.loc[metadata['Modality']=='SEG', 'Subject ID'].unique().tolist()
seg_ids = [id_ for id_ in seg_ids
           if len(metadata[(metadata['Subject ID']==id_)&(metadata['Modality']=='CT')])==1]

# create batches of patient_ids to
# process image data in parallel
n_batches = config["n_batches"]
n_ids = (len(seg_ids)//n_batches) + 1
seg_ids_dict = {
    str(n+1): seg_ids[n*n_ids:(n+1)*n_ids] for n in range(n_batches)
}


# PIPELINE
# --------

rule all:
    "The default rule"
    input:
        expand("results/{data}_model_metrics.csv", data=[d for d in config["model_data"]]),
        expand("results/roc_plots/{model}_{data}_model_roc.png",
               model=["lreg", "mlp"],
               data=[d for d in config["model_data"]]),
        expand("results/feature_plots/lreg_{data}_model_coefs.png",
               data=[d for d in config["model_data"]]),
        expand("results/feature_plots/mlp_{data}_model_shap.png",
               data=[d for d in config["model_data"]])

rule process_clinical:
    "Process raw clinical data"
    output: 
        expand("data/clean/clinical{i}.csv", i = [1, 2]),
        "data/clean/clinical_joined.csv"
    log:
        "logs/snakemake/process_clinical_data.log"
    shell: """
    mkdir -p logs/snakemake 2>&1 | tee {log}
    echo Starting to process clinical data 2>&1 | tee -a {log}
    date 2>&1 | tee -a {log}
    mkdir -p data/clean 2>&1 | tee -a {log}
    python code/process_clinical_data.py 2>&1 | tee -a {log}
    echo Finished processing clinical data 2>&1 | tee -a {log}
    date 2>&1 | tee -a {log}
    """

for batch, ids in seg_ids_dict.items():
    rule:
        name:
            f"process_images_batch_{batch}"
        params:
            batch = f"{batch}",
            ids = ",".join([str(id) for id in ids])
        input:
            #expand("data/metadata{i}.csv", i=[1,2]),
            "data/clean/clinical_joined.csv"
        output:
            f"data/clean/image_features_{batch}.csv"
        log:
            f"logs/snakemake/process_image_data_{batch}.log"
        shell: """
        echo "Starting to process the images for patients in batch {params.batch}" 2>&1 | tee {log}
        date 2>&1 | tee -a {log}
        python code/process_image_data.py {params.batch} "{params.ids}" 2>&1 | tee -a {log}
        echo "Processing of images for patients in batch {params.batch} complete" 2>&1 | tee -a {log}
        date 2>&1 | tee -a {log}
        """

for data in config["model_data"]:
    rule:
        name: f"develop_models_for_{data}"
        params: 
            data = f"{data}",
            random_seed = f"{config["random_seed"]}",
            k_folds = f"{config["k_folds"]}",
            n_batches = f"{n_batches}",
            test_split = f"{config["test_split"]}"
        input:
            "data/clean/clinical1.csv",
            "data/clean/clinical2.csv",
            "data/clean/clinical_joined.csv",
            expand("data/clean/image_features_{i}.csv", i=range(1, n_batches+1))
        output: 
            expand("data/models/{data}/{type}.csv", data=data, type=["train", "test"]),
            expand("results/models/{model}_{data}_model.pkl", model=["lreg", "mlp"], data=data)
        log:
            f"logs/snakemake/train_{data}_models.log"
        shell: """
        echo "Begin developing the models for {params.data} data" 2>&1 | tee {log}
        date 2>&1 | tee -a {log}
        mkdir -p data/models/{params.data} 2>&1 | tee -a {log}
        mkdir -p results/models 2>&1 | tee -a {log}
        python code/train_models.py {params.data} {params.random_seed} {params.k_folds} {params.n_batches} {params.test_split} 2>&1 | tee -a {log}
        echo "Finished developing models for {params.data} data" 2>&1 | tee -a {log}
        date 2>&1 | tee -a {log}
        """

for data in config["model_data"]:
    rule:
        name: f"evaluate_models_trained_on_{data}"
        params: 
            data = f"{data}"
        input:
            f"data/models/{data}/test.csv",
            expand("results/models/{model}_{data}_model.pkl", model=["lreg", "mlp"], data=data)
        output: 
            f"results/{data}_model_metrics.csv",
            expand("results/roc_plots/{model}_{data}_model_roc.png",
                   model=["lreg", "mlp"],
                   data=data),
            f"results/feature_plots/lreg_{data}_model_coefs.png",
            f"results/feature_plots/mlp_{data}_model_shap.png"
        log:
            f"logs/snakemake/evaluate_{data}_models.log"
        shell: """
        echo "Begin evaluating the models for the {params.data} data" 2>&1 | tee {log}
        date 2>&1 | tee -a {log}
        python code/evaluate_models.py {params.data} 2>&1 | tee -a {log}
        echo "Finished evaluating the models for the {params.data} data" 2>&1 | tee -a {log}
        date 2>&1 | tee -a {log}
        """

rule clean:
    "Clean up"
    shell: """
    if [ -d logs ]; then
      echo "Removing directory logs/snakemake"
      rm -r logs/snakemake
    else
      echo directory logs/snakemake does not exist
    fi
    if [ -d data/clean ]; then
      echo "Removing directory data/clean"
      rm -r data/clean
    else
      echo directory code/clean does not exist
    fi
    if [ -d data/models ]; then
      echo "Removing directory data/models"
      rm -r data/models
    else
      echo directory code/models does not exist
    fi
    if [ -d results ]; then
      echo "Removing directory results"
      rm -r results
    else
      echo directory results does not exist
    fi
    """
