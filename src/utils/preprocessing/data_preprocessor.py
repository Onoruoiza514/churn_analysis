import pandas as pd
import numpy as np


from src.utils.loggers import logger
from src.utils.exception.exceptions  import CustomException
import sys
import os

raw_data_path = "data/raw/Telecos_customer_churn.csv"

def load_raw_data():
    try:
        logger.info("Loading raw data")
        df = pd.read_csv(raw_data_path)
        logger.info(f"Data loaded with shape {df.shape}")
        return df
    except Exception as e:
        raise CustomException(e, sys)

import pandas as pd
import sys
from src.utils.loggers import logger
from src.utils.exception.exceptions import CustomException


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and encodes the Telco Customer Churn dataset.
    """
    try:
        logger.info("Starting data cleaning process")
        df = df.copy()

        # 1. Drop customerID
        if "customerID" in df.columns:
            df.drop(columns=["customerID"], inplace=True)
            logger.info("Dropped customerID column")

        # 2. Strip whitespace from object columns
        obj_cols = df.select_dtypes(include="object").columns
        for col in obj_cols:
            df[col] = df[col].str.strip()

        # 3. Convert TotalCharges to numeric
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # 4. Handle missing TotalCharges
        missing_before = df["TotalCharges"].isna().sum()
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
        logger.info(f"Filled {missing_before} missing TotalCharges values")

        # 5. Convert Churn to binary
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

        # 6. Ensure SeniorCitizen is int
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

        # 7. One-hot encode categorical columns (excluding target)
        categorical_cols = df.select_dtypes(include="object").columns.tolist()
        logger.info(f"Encoding categorical columns: {categorical_cols}")

        df_encoded = pd.get_dummies(
            df,
            columns=categorical_cols,
            drop_first=True
        )

        # 8. Convert booleans to int
        bool_cols = df_encoded.select_dtypes(include="bool").columns
        df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

        logger.info(f"Data cleaning completed. Final shape: {df_encoded.shape}")
        return df_encoded

    except Exception as e:
        logger.error("Error during data cleaning")
        raise CustomException(e, sys)


def save_cleaned_data(
    df_encoded,
    output_path: str = "data/processed/cleaned_telco_data.csv"
):
    """
    Saves our cleaned data to the processed directory.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_encoded.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")

    except Exception as e:
        logger.error("Failed to save processed data")
        raise CustomException(e, sys)


def main():
    df_raw = load_raw_data()
    df_cleaned = clean_data(df_raw)
    save_cleaned_data(df_cleaned)
if __name__ == "__main__":
    main()
