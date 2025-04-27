# MLDM Summative Assessment

## Overview
The aim of this project to develop machine learning models to predict mortality for lung cancer patients. To achieve this, we process clinical and CT image data, train L1-penalized logistic regression (LASSO) and multilayer perceptron (MLP) models and evaluate the models.

All tasks are designed to run on the high performance computer (HPC) BlueCrystal as part of a SnakeMake pipeline.


## Data
The data for this project include two sources of clinical data and CT images for most patients.

The first clinical dataset includes 420 observations and 7 features. The second clinical dataset includes 180 observations and 26 features. There are 421 patients with CT images overall which we extract 19 features from.

Furthermore, the clinical datasets are combined to create a 'clinical joined' dataset which has 600 observations and only includes the 7 features which are common between the two clinical datasets to avoid imputing roughly two thirds of datapoints for the other 19 features that are in the second clinical dataset.

A 'full' dataset is also created by joining the image features onto the clinical joined dataset which results in a dataset with 600 observations and 26 features.


## Methods
### Packages
The data are processed and models are trained and evaluated using python version 3.12.0 as with the following packages: pandas, pydicom, scikit-image, scikit-learn, matplotlib and shap.

### Data processing
The general procedure for processing the clinical data is

1. Clean column names
2. Rename columns
3. Categorise data
4. Feature engineering (including deriving the target variable 'death within 2 years of diagnosis')

To process the CT images we extract tumour morphological features, gray level co-occurance features and intensity features using grayscale and segmented images.

The process_clinical_data.py and process_image_data.py scripts are responsible for loading and processing the data.

### Model development
The train_models.py script is responsible for developing the models.

#### Pre-processing
First, we split the data into a 80% training set and 20% test set.

We pre-process the data depending on feature type. The way in which we pre-process the data is outlined below:

* Nominal data: Impute missing data using the mode and then one hot encoding.
* Ordinal data: Impute missing data with the value -1.
* Numerical data: Impute missing data using the median and then scale using min-max normalisation.

#### Models
We develop an L1-penalised logistic regression (LASSO) model and a multilayer perceptron (MLP) model on each of the 5 datasets.

Furthermore, we tune hyperparameters for each of the models using grid search 5 fold cross-validation, and train the models using the hyperparameters that maximise AUC on the full training set.


### Model evaluation
For each model-dataset combination we extract the following metrics using the testing set: AUC, Weighted Accuracy, F1 Score. We also plot the ROC curve for each model-dataset combination. Finally, we plot the 10 coefficients with the largest absolute values for the LASSO models and the 10 features with the largest SHAP values for the MLP models.

The evaluate_models.py script is responsible for evaluating the models.


## More Information
### Project
This project aims to understand which features are most predictive of mortality among lung cancer patients and compare a 'simple' and 'complex' machine learning model in terms of performance.

### Tech Stack
Environment management:
* conda
Programming languages:
* Python (version 3.12.0)
- snakemake
- pandas
- pydicom
- scikit-image
- scikit-learn
- matplotlib
- shap
High performance computing:
* BlueCrystal
* Slurm

### Instructions
#### One time set-up
Before running the pipeline on the HPC for the first time you must follow some set-up steps.
First, clone this git repository to a suitable area of the login node in the HPC by running the following command:
```
git clone https://github.com/RhodriDavidJohn/mldm-summative.git
```
Before starting the set-up, ensure you have cloned the git repository and you are in the root directory of the repository.

##### Set up Slurm profile to run pipeline
You will only need to set up the Slurm profile once.
To do this run the following command to submit a Slurm job to set up your Slurm profile:
```
sbatch code/setup/hpc_setup_job.sh
```
You can check the progress of the job by running the command:
```
sacct -j {job-id} -X
```
Note, you will need to replace {job-id} with the job ID given by the first command.

##### Set up Conda environment
First you will need to ensure that you have correctly set up Conda on the HCP.
Then you can create and activate the environment for the pipeline by running the following commands, again ensuring you are in the root directory of the repository:
```
source ~/initConda.sh
conda env create -n mldm-env --file conda_env.yml
conda activate mldm-env
```
Note, you will only need to create the environment once.
After you have created the environment, if you ever want to activate it, simply run the following commands:
```
source ~/initConda.sh
conda activate mldm-env
```

#### Running the pipeline
If you have successfully completed the HCP set-up steps above and the Conda environment is active, then you can run the pipeline.


##### Pipeline commands
To run the full pipeline, run the following command:
```
snakemake --executor slurm --profile mldm_slurm_profile
```

If you want to clean the repository (i.e., remove all data [not the metadata], results, and logs), run the following command:
```
snakemake --executor slurm --profile mldm_slurm_profile clean
```
