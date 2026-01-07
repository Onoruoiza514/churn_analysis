## Telco Customer Churn Analysis & Prediction
Project Overview
This project delivers an end-to-end customer churn analytics and prediction system built on the Telco Customer Churn dataset.
It covers the full data science lifecycle:
    * Exploratory analysis and feature design (Notebook)
    * Production-ready modular Python codebase
    * Reusable ML pipeline with logging and exception handling
    * Model training and optimization (ROC-AUC ≈ 85%)
    * Interactive Tableau dashboard for business insights

The project is structured to meet real-world, production, and portfolio standards.
Objectives
    * Understand customer churn drivers
    * Build a robust churn prediction model
    * Modularize notebook logic into maintainable scripts
    * Enable repeatable pipeline execution
    * Provide decision-ready visual analytics using Tableau

Tech Stack
    * Languages & Libraries
    * Python 3.10+
    * pandas, numpy , scikit-learn , XGBoost , matplotlib, seaborn , Tableau (Dashboarding).

Engineering & MLOps Concepts
    * Modular architecture
    * Logging & custom exceptions
    * Reusable pipelines
    * Train-test reproducibility
    * Model persistence (pickle)

Workflow Summary
1. Data Preprocessing

Implemented in data_preprocessor.py:
- Dropped non-predictive identifiers
- Handled missing values
- Converted target (Churn) to binary
- One-hot encoded categorical variables
- Saved cleaned data to data/processed
- All steps mirror the original notebook logic.

2. Feature Engineering
- Implemented in feature_engineering.py:
- Numerical feature consistency checks
- Feature transformations used in modelling
- Output saved as modelling-ready dataset
- This ensures training and inference parity.

3. Data Splitting
- Implemented in data_splitter.py:
- Train / Test split
- Explicit target separation
- Reproducible random state
- Centralized for reuse across models

4. Model Training & Optimization
- Implemented in train.py.
  Models Used
     - Logistic Regression
     - Random Forest
     - XGBoost (primary model)
- Key Techniques Include:
* Standard scaling where required
* Class imbalance handling (scale_pos_weight)
* Hyperparameter tuning with GridSearchCV / RandomizedSearchCV using ROC-AUC
* Best Model Performance : XGBoost ROC-AUC ≈ 0.85
* Trained models are persisted in /models.

5. Pipeline Execution
* The entire workflow can be executed using:
* python run_pipeline.py
This runs:
    - Preprocessing
    - Feature engineering
    - Train-test split
    - Model training
    - Model saving

6. Logging & Error Handling
- Centralized logging system
- Custom exception class with file & line tracing
- Logs stored in /logs
- Pipeline failures are traceable and debuggable

7. Tableau Dashboard
Purpose:
    - Provide business-ready insights beyond the ML model.
    - Data Used
    - Raw dataset (to retain Customer ID and categorical richness
    - Visuals Included
    - Churn distribution overview
    - Payment method distribution (single pie)
    - Contract type churn comparison
    - Tenure vs churn patterns
    - Donut charts for categorical breakdowns

Author
Abdulfaatihi Onoruoiza Tijani
Data Scientist | ML Engineer | AI Engineer
