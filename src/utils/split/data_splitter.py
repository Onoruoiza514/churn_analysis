import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.loggers import logger
from src.utils.exception.exceptions import CustomException

data_path = "data/modelling/modelling_ready_telco_data.csv"
df = pd.read_csv(data_path)
def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    This script splits dataset into train and test sets.
    """

    try:
        logger.info("Starting data splitting")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        logger.info(
            f"Data split completed | "
            f"X_train: {X_train.shape}, X_test: {X_test.shape}"
        )

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error("Error during data splitting")
        raise CustomException(e, sys)

if __name__ == "__main__":
    from src.utils.preprocessing.data_preprocessor import load_raw_data, clean_data
    from src.utils.features.feature_engineering import create_features

    df = load_raw_data()
    df = clean_data(df)
    df = create_features(df)

    X_train, X_test, y_train, y_test = split_data(df, target_col="Churn")

    print(X_train.shape, X_test.shape)