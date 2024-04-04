# Azure Machine Learning Pipeline Template for Classification Task

- ML pipeline template for my project of classification task
- Used Azure Machine Learning Studio to develop pipelines

## Directory Structure
```
├── env
│   ├── conda_env.yml (Install packages that will be used for pipeline execution)
│   └── register_env.py (Register an environment on a workspace)
├── pipeline
│   ├── read_data.py (Read data from our DB and register to ML Studio)
│   ├── train.py (Train models)
│   ├── register_model.py (Register our trained models to ML Studio)
│   ├── predict.py (Predict models)
│   └── upload_db.py (Upload data to our DB)
├── utils
│   ├── connection.py (SQL commands for connection to DB)
│   ├── label_encoding.py (Label encoding)
│   └── model (XGBoost)
├── .gitlab-ci.yml (Run codes triggered from Gitlab schedule)
├── azure_ml_pipeline.py (Run ML pipeline)
├── config.ini (Config file for settings)
├── logger.py (Logging)
└── README.md
```
