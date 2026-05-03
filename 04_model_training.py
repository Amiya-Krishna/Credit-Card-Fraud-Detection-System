"""
PHASE 5: MODEL TRAINING & HYPERPARAMETER TUNING
"""

import os
import pandas as pd
import numpy as np
import joblib
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier
import optuna

# =========================
# INIT PROJECT (FIXES ALL FOLDER ERRORS)
# =========================
def init_project():
    folders = ['data', 'outputs', 'models', 'logs']
    for f in folders:
        os.makedirs(f, exist_ok=True)

init_project()

# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =========================
# LOAD DATA
# =========================
def load_processed_data():
    logger.info("📂 Loading preprocessed data...")
    
    X_train = joblib.load('data/X_train_scaled.joblib')
    X_val = joblib.load('data/X_val_scaled.joblib')
    y_train = joblib.load('data/y_train.joblib')
    y_val = joblib.load('data/y_val.joblib')
    
    return X_train, X_val, y_train, y_val


# =========================
# BASELINE MODELS
# =========================
def train_baseline_models(X_train, X_val, y_train, y_val):

    results = {}

    # Logistic Regression
    logit = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1)
    logit.fit(X_train, y_train)

    proba_logit = logit.predict_proba(X_val)[:, 1]
    results['LogisticRegression'] = {
        'model': logit,
        'proba': proba_logit,
        'pr_auc': average_precision_score(y_val, proba_logit),
        'roc_auc': roc_auc_score(y_val, proba_logit),
    }

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight='balanced',
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    proba_rf = rf.predict_proba(X_val)[:, 1]
    results['RandomForest'] = {
        'model': rf,
        'proba': proba_rf,
        'pr_auc': average_precision_score(y_val, proba_rf),
        'roc_auc': roc_auc_score(y_val, proba_rf),
    }

    return results


# =========================
# OPTUNA TUNING
# =========================
def tune_xgboost_with_optuna(X_train, X_val, y_train, y_val, n_trials=30):

    pos_weight = (len(y_train) - y_train.sum()) / max(1, y_train.sum())
    logger.info(f"scale_pos_weight: {pos_weight:.2f}")

    def objective(trial):

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 6),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'scale_pos_weight': pos_weight,
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0,
        }

        model = XGBClassifier(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )

        proba = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, proba)

    study = optuna.create_study(
        direction="maximize",
        study_name="xgboost_tuning",
        storage="sqlite:///optuna.db",
        load_if_exists=True
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        catch=(Exception,)
    )

    return study.best_params, pos_weight


# =========================
# FINAL MODEL
# =========================
def train_final_model(X_train, X_val, y_train, y_val, best_params, pos_weight):

    best_params.update({
        'scale_pos_weight': pos_weight,
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
    })

    model = XGBClassifier(**best_params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )

    proba = model.predict_proba(X_val)[:, 1]

    logger.info(f"PR-AUC: {average_precision_score(y_val, proba):.4f}")
    logger.info(f"ROC-AUC: {roc_auc_score(y_val, proba):.4f}")

    return model, proba


# =========================
# THRESHOLD OPTIMIZATION
# =========================
def find_optimal_threshold(y_val, proba):

    best_threshold = 0.5
    best_cost = float('inf')

    for t in np.linspace(0.01, 0.99, 99):
        y_pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

        cost = (fn * 5000) + (fp * 50)

        if cost < best_cost:
            best_cost = cost
            best_threshold = t

    logger.info(f"Best threshold: {best_threshold:.3f}")
    logger.info(f"Cost: ₹{best_cost:,.0f}")

    return best_threshold


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    X_train, X_val, y_train, y_val = load_processed_data()

    baseline = train_baseline_models(X_train, X_val, y_train, y_val)

    best_params, pos_weight = tune_xgboost_with_optuna(
        X_train, X_val, y_train, y_val
    )

    model, proba = train_final_model(
        X_train, X_val, y_train, y_val, best_params, pos_weight
    )

    threshold = find_optimal_threshold(y_val, proba)

    # SAVE
    joblib.dump(model, 'models/xgboost_final.joblib')

    bundle = {
        'model': model,
        'threshold': float(threshold),
        'params': best_params
    }

    joblib.dump(bundle, 'models/fraud_model_bundle.joblib')

    logger.info("✅ Training complete & models saved!")