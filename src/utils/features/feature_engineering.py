import pandas as pd
import sys
import os

from src.utils.loggers import logger
from src.utils.exception.exceptions import CustomException

cleaned_df_path = "data/processed/cleaned_telco_data.csv"
cleaned_df = pd.read_csv(cleaned_df_path)
def create_features(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes cleaned telco dataframe and creates model-ready features.
    Mirrors the feature engineering notebook.
    """
    try:
        logger.info("Starting feature engineering process")
        df_fe = cleaned_df.copy()

        # 1. Tenure groups
        df_fe["tenure_group"] = pd.cut(
            df_fe["tenure"],
            bins=[0, 12, 24, 48, 72],
            labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"]
        )

        # Then we use one hot encoding to convert to what machine learning models understand as they do not understand category data type
        df_new = pd.get_dummies(df_fe, columns=["tenure_group"], drop_first=True)
        logger.info("Created tenure_group feature and encoded successsfully......")

        # 2. Contract strength
        contract_cols = ["Contract_One year", "Contract_Two year"]
        for col in contract_cols:
            if col not in df_new.columns:
                df_new[col] = 0
        df_new["contract_strength"] = df_new["Contract_Two year"].astype(int) * 2 + df_new["Contract_One year"].astype(int)
        logger.info("Created contract_strength feature")

        # 3. Has internet (binary flag)
        if "InternetService_Fiber optic" not in df_new.columns:
            df_new["InternetService_Fiber optic"] = 0
        if "InternetService_No" not in df_new.columns:
            df_new["InternetService_No"] = 0
        df_new["has_internet"] = (
                df_new["InternetService_Fiber optic"] | df_new["InternetService_No"]
        ).astype(int)
        logger.info("Created has_internet feature")

        # 4. Service count (number of enabled services)
        service_cols = [
            "PhoneService_Yes", "MultipleLines_Yes", "OnlineSecurity_Yes",
            "OnlineBackup_Yes", "DeviceProtection_Yes", "TechSupport_Yes",
            "StreamingTV_Yes", "StreamingMovies_Yes"
        ]
        for col in service_cols:
            if col not in df_new.columns:
                df_new[col] = 0
        df_new["service_count"] = df_new[service_cols].sum(axis=1)
        logger.info("Created service_count feature")

        # 5. Is long-term customer (tenure >= 24 months)
        df_new["is_long_term"] = (df_new["tenure"] >= 24).astype(int)
        logger.info("Created is_long_term feature")

        # 6. Avg monthly charge (TotalCharges / tenure)
        df_new["avg_monthly_charge"] = df_new["TotalCharges"] / df_new["tenure"].replace(0, 1)
        df_new["high_monthly_charge"] = (df_new["MonthlyCharges"] > df_new["MonthlyCharges"].median()).astype(int)
        logger.info("Created avg_monthly_charge and high_monthly_charge features")

        tenure_cols = ['tenure_group_1-2yr', 'tenure_group_2-4yr', 'tenure_group_4-6yr']
        df_new[tenure_cols] = df_new[tenure_cols].astype(int)

        logger.info(f"Feature engineering completed. Shape: {df_new.shape}")
        return df_new

    except Exception as e:
        logger.error("Error occurred during feature engineering")
        raise CustomException(e, sys)


def save_features(df_new: pd.DataFrame, path="data/modelling/modelling_ready_telco_data.csv"):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df_new.to_csv(path, index=False)
        logger.info(f"Feature engineered data saved to {path}")
    except Exception as e:
        logger.error("Failed to save feature engineered data")
        raise CustomException(e, sys)

if __name__ == "__main__":
    from src.utils.preprocessing.data_preprocessor import load_raw_data, clean_data


    df_raw = load_raw_data()
    df_cleaned = clean_data(df_raw)
    df_features = create_features(df_cleaned)
    save_features(df_features)

