# DVC and DagsHub Exam Submission

This repository implements the solution architecture for the exam using DVC for data and model versioning and DagsHub for remote storage and collaboration.

## Repository Structure

The repository is organized as follows:

```bash       
├── examen_dvc          
│   ├── data       
│   │   ├── processed   # Contains processed data (train/test splits, normalized data)
│   │   └── raw         # Contains raw data (raw.csv and raw_new.csv generated by wrangle)
│   ├── metrics         # Contains evaluation metrics (scores.json)
│   ├── models          # Contains model artifacts and parameters (trained_model.joblib, best_params.pkl)
│   ├── src             # Contains source scripts for data processing and modeling:
│   │   ├── data/       # (wrangle.py, split_data.py, normalize_data.py)
│   │   └── models/     # (grid_search.py, train_model.py, evaluate_model.py)
│   ├── dvc.yaml        # DVC pipeline definition
│   ├── dvc.lock        # DVC lock file with stage outputs and checksums
│   ├── .dvc/           # DVC configuration folder
│   ├── .gitignore      
│   └── README.md       
     
```

## How to Set Up the Repository

### Download the Data:
I downloaded the raw dataset from the following link and placed it in the data/raw/ directory:
https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv

### Set up *.py-files
I set up the *.py-files and tested them manually.

### Fork and Clone:
I forked the repository on GitHub and cloned it locally to work on my own solution: https://github.com/Anomalisaa/examen-dvc/

I also cloned the repo to my Dagshub account:https://dagshub.com/Anomalisaa/examen-dvc

### requirements.txt file
Set up a requirements.txt file.

### Remote Collaboration:
I added https://dagshub.com/licence.pedago as a collaborator with read-only access for grading.

## DVC pipeline
I set up a DVC pipeline that reproduced the manual workflow by utilizing the scripts.

# Credentials
name: Isabell Gurstein

email: isabell.gurstein@realcore.de 

dagshub: https://dagshub.com/Anomalisaa/examen-dvc

github: https://github.com/Anomalisaa/examen-dvc
