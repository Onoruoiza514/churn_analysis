import sys
import pandas as pd

from src.utils.loggers import logger
from src.utils.exception.exceptions import CustomException

from src.utils.preprocessing.data_preprocessor import (
    load_raw_data,
    clean_data,
    save_cleaned_data
)

from src.utils.features.feature_engineering import create_features
from src.utils.split.data_splitter import split_data
from src.utils.models.train import (
    train_baseline_models,
    tune_random_forest,
    tune_xgboost,
    evaluate_and_save
)


def run_pipeline():
    try:
        logger.info("========== PIPELINE STARTED ==========")

        # 1. Load raw data
        df_raw = load_raw_data()

        # 2. Clean data
        df_clean = clean_data(df_raw)
        save_cleaned_data(df_clean)

        # 3. Feature engineering
        df_fe = create_features(df_clean)

        modelling_path = "data/modelling/modelling_ready_telco_data.csv"
        df_fe.to_csv(modelling_path, index=False)
        logger.info(f"Modelling-ready data saved at {modelling_path}")

        # 4. Train-test split
        X_train, X_test, y_train, y_test = split_data(
            df=df_fe,
            target_col="Churn"
        )

        # 5. Baseline models
        train_baseline_models(X_train, X_test, y_train, y_test)

        # 6. Tuned models
        best_rf = tune_random_forest(X_train, y_train)
        best_xgb = tune_xgboost(X_train, y_train)

        # 7. Final evaluation & saving
        evaluate_and_save(best_rf, X_test, y_test, "random_forest")
        evaluate_and_save(best_xgb, X_test, y_test, "xgboost")

        logger.info("========== PIPELINE COMPLETED SUCCESSFULLY ==========")

    except Exception as e:
        logger.error("PIPELINE FAILED")
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_pipeline()
