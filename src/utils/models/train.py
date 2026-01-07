import sys
import os
import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier

from src.utils.loggers import logger
from src.utils.exception.exceptions import CustomException
from src.utils.split.data_splitter import split_data


MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def train_baseline_models(X_train, X_test, y_train, y_test):
    try:
        logger.info("Training baseline models")

        scores = {}

        #Logistic Regression(scaled) only baseline model logistic regression scaled
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train_scaled, y_train)

        y_probs_lr = lr.predict_proba(X_test_scaled)[:, 1]
        lr_roc_auc = roc_auc_score(y_test, y_probs_lr)

        scores["logistic_regression"] = lr_roc_auc
        logger.info(f"logistic_regression ROC-AUC: {lr_roc_auc:.4f}")

        #Random Forest(unscaled)
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)

        y_probs_rf = rf.predict_proba(X_test)[:, 1]
        rf_roc_auc = roc_auc_score(y_test, y_probs_rf)

        scores["random_forest"] = rf_roc_auc
        logger.info(f"random_forest ROC-AUC: {rf_roc_auc:.4f}")

        return scores

    except Exception as e:
        raise CustomException(e, sys)


def tune_random_forest(X_train, y_train):
    try:
        logger.info("Starting RandomForest RandomizedSearchCV")

        rf = RandomForestClassifier(random_state=42)

        param_grid_rf = {
            "n_estimators": [200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }

        rf_random = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_grid_rf,
            n_iter=20,
            scoring="roc_auc",
            cv=5,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )

        rf_random.fit(X_train, y_train)

        logger.info(f"Best RF params: {rf_random.best_params_}")
        logger.info(f"Best RF ROC-AUC: {rf_random.best_score_:.4f}")

        return rf_random.best_estimator_

    except Exception as e:
        raise CustomException(e, sys)


def tune_xgboost(X_train, y_train):
    try:
        logger.info("Starting XGBoost GridSearchCV")

        xgb = XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss"
        )

        param_grid_xgb = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "scale_pos_weight": [
                (y_train == 0).sum() / (y_train == 1).sum()
            ]
        }

        xgb_grid = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid_xgb,
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
            verbose=2
        )

        xgb_grid.fit(X_train, y_train)

        logger.info(f"Best XGB params: {xgb_grid.best_params_}")
        logger.info(f"Best XGB ROC-AUC: {xgb_grid.best_score_:.4f}")

        return xgb_grid.best_estimator_

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_and_save(model, X_test, y_test, model_name):
    try:
        y_probs = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_probs)

        logger.info(f"{model_name} FINAL ROC-AUC: {roc_auc:.4f}")

        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        logger.info(f"Model saved at {model_path}")

        return roc_auc

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        logger.info("Starting model training pipeline")
        df = pd.read_csv("data/modelling/modelling_ready_telco_data.csv")

        X_train, X_test, y_train, y_test = split_data(
            df=df,
            target_col="Churn"
        )

        # Baseline models
        train_baseline_models(X_train, X_test, y_train, y_test)

        # Tuned models
        best_rf = tune_random_forest(X_train, y_train)
        best_xgb = tune_xgboost(X_train, y_train)

        # Final evaluation & saving
        evaluate_and_save(best_rf, X_test, y_test, "random_forest")
        evaluate_and_save(best_xgb, X_test, y_test, "xgboost")

        logger.info("Model training pipeline completed successfully")

    except Exception as e:
        logger.error("Model training pipeline failed")
        raise CustomException(e, sys)
