#!/usr/bin/env python
# -*- coding: utf-8 -*-

# MODELADO CATBOOST

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

# Preprocesamiento
from sklearn.model_selection import train_test_split, StratifiedKFold

# Modelo
from catboost import CatBoostClassifier

# Métricas
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)

# Encoders
from category_encoders import TargetEncoder

# Optimización
import optuna

# MLflow
import mlflow
import mlflow.sklearn

# ==============================================================================
# CONFIGURACIÓN DE RUTAS
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PROCESSED_PATH      = PROJECT_ROOT / "data" / "processed" / "preprocessed_data.csv"
OUTPUT_DIR_FIGURES       = PROJECT_ROOT / "outputs" / "figures" / "modelado" / "CatBoost"
OUTPUT_DIR_MODELS        = PROJECT_ROOT / "outputs" / "models"  / "CatBoost"
OUTPUT_DIR_MODELS_GLOBAL = PROJECT_ROOT / "outputs" / "models"
MLRUNS_DIR               = PROJECT_ROOT / "mlruns"

# Semilla global
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Estilo de visualización
plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.4f}".format)

# ==============================================================================
# DEFINICIÓN DE VARIABLES POR FASE TEMPORAL
# ==============================================================================

TARGET = "target_binario"

VARS_BINARIAS_T0 = [
    "daytimeevening_attendance",
    "displaced",
    "educational_special_needs",
    "gender",
    "scholarship_holder",
    "international",
    "is_single",
]

VARS_BINARIAS_T1 = [
    "debtor",
    "tuition_fees_up_to_date",
]

VARS_NUMERICAS_T0 = [
    "age_at_enrollment",
    "admission_grade",
    "previous_qualification_grade",
]

VARS_NUMERICAS_T1 = [
    "curricular_units_1st_sem_credited",
    "curricular_units_1st_sem_enrolled",
    "curricular_units_1st_sem_evaluations",
    "curricular_units_1st_sem_approved",
    "curricular_units_1st_sem_grade",
    "curricular_units_1st_sem_without_evaluations",
    "unemployment_rate",
    "inflation_rate",
    "gdp",
]

VARS_NUMERICAS_T2 = [
    "curricular_units_2nd_sem_credited",
    "curricular_units_2nd_sem_enrolled",
    "curricular_units_2nd_sem_evaluations",
    "curricular_units_2nd_sem_approved",
    "curricular_units_2nd_sem_grade",
    "curricular_units_2nd_sem_without_evaluations",
]

# CatBoost maneja estas variables nativamente como categóricas (string)
VARS_CATEGORICAS_AGRUPADAS_T0 = [
    "application_mode_risk",
    "previous_qualification_risk",
    "mothers_qualification_level",
    "fathers_qualification_level",
    "mothers_occupation_level",
    "fathers_occupation_level",
]

VARS_TARGET_ENCODING_T0 = ["course"]

VARS_ORDINALES_T0 = ["application_order"]

VARS_T0 = (
    VARS_BINARIAS_T0
    + VARS_NUMERICAS_T0
    + VARS_CATEGORICAS_AGRUPADAS_T0
    + VARS_TARGET_ENCODING_T0
    + VARS_ORDINALES_T0
)

VARS_T1 = VARS_T0 + VARS_BINARIAS_T1 + VARS_NUMERICAS_T1

VARS_T2 = VARS_T1 + VARS_NUMERICAS_T2


# ==============================================================================
# FUNCIONES DE PREPROCESAMIENTO
# ==============================================================================

def obtiene_variables_por_fase(fase: str) -> dict:

    if fase == "T0":
        return {
            "binarias":            VARS_BINARIAS_T0,
            "numericas":           VARS_NUMERICAS_T0 + VARS_ORDINALES_T0,
            "categoricas_nativas": VARS_CATEGORICAS_AGRUPADAS_T0,
            "categoricas_te":      VARS_TARGET_ENCODING_T0,
            "all":                 VARS_T0,
        }
    elif fase == "T1":
        return {
            "binarias":            VARS_BINARIAS_T0 + VARS_BINARIAS_T1,
            "numericas":           VARS_NUMERICAS_T0 + VARS_ORDINALES_T0 + VARS_NUMERICAS_T1,
            "categoricas_nativas": VARS_CATEGORICAS_AGRUPADAS_T0,
            "categoricas_te":      VARS_TARGET_ENCODING_T0,
            "all":                 VARS_T1,
        }
    elif fase == "T2":
        return {
            "binarias":            VARS_BINARIAS_T0 + VARS_BINARIAS_T1,
            "numericas":           VARS_NUMERICAS_T0 + VARS_ORDINALES_T0 + VARS_NUMERICAS_T1 + VARS_NUMERICAS_T2,
            "categoricas_nativas": VARS_CATEGORICAS_AGRUPADAS_T0,
            "categoricas_te":      VARS_TARGET_ENCODING_T0,
            "all":                 VARS_T2,
        }
    else:
        raise ValueError(f"Fase no válida: {fase}. Usar 'T0', 'T1' o 'T2'.")


def preprocesamiento_catboost(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    fase: str,
) -> tuple:

    variables_fase = obtiene_variables_por_fase(fase)

    X_train_fase = X_train[variables_fase["all"]].copy()
    X_test_fase  = X_test[variables_fase["all"]].copy()

    # ------------------------------------------------------------------
    # 1. Target Encoding para 'course'
    # ------------------------------------------------------------------
    te = TargetEncoder(cols=variables_fase["categoricas_te"], smoothing=0.3)

    for col in variables_fase["categoricas_te"]:
        X_train_fase[col + "_encoded"] = te.fit_transform(X_train_fase[[col]], y_train)[col]
        X_test_fase[col + "_encoded"]  = te.transform(X_test_fase[[col]])[col]
        X_train_fase = X_train_fase.drop(columns=[col])
        X_test_fase  = X_test_fase.drop(columns=[col])

    # ------------------------------------------------------------------
    # 2. Convertir categóricas a string (CatBoost las requiere así)
    # ------------------------------------------------------------------
    for col in variables_fase["categoricas_nativas"]:
        X_train_fase[col] = X_train_fase[col].astype(str)
        X_test_fase[col]  = X_test_fase[col].astype(str)

    # ------------------------------------------------------------------
    # 3. Índices de columnas categóricas para CatBoost
    # ------------------------------------------------------------------
    cat_features_idx   = [X_train_fase.columns.get_loc(c)
                          for c in variables_fase["categoricas_nativas"]]
    cat_features_names = variables_fase["categoricas_nativas"]

    feature_names = X_train_fase.columns.tolist()
    preprocessors = {
        "target_encoder":     te,
        "feature_names":      feature_names,
        "cat_features_idx":   cat_features_idx,
        "cat_features_names": cat_features_names,
    }

    return X_train_fase, X_test_fase, feature_names, preprocessors


# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def _calcula_class_weights(y_train: pd.Series) -> dict:
    """Calcula class_weights = {0: 1.0, 1: n_neg/n_pos}."""
    spw = float((y_train == 0).sum() / (y_train == 1).sum())
    return {0: 1.0, 1: spw}


def _grafica_curva_perdida(
    train_losses: list,
    val_losses: list,
    fase: str,
    cv_folds: int,
    tag_optimizado: bool,
    output_dir: Path = OUTPUT_DIR_FIGURES,
) -> None:
    """
    Genera y guarda la curva de pérdida (Logloss) vs iteraciones con bandas ± std.
    """
    min_len   = min(len(l) for l in train_losses)
    train_arr = np.array([l[:min_len] for l in train_losses])
    val_arr   = np.array([l[:min_len] for l in val_losses])

    train_mean = train_arr.mean(axis=0)
    train_std  = train_arr.std(axis=0)
    val_mean   = val_arr.mean(axis=0)
    val_std    = val_arr.std(axis=0)
    iterations = range(1, min_len + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterations, train_mean, label="Train Loss",      color="#3498DB", linewidth=2)
    ax.fill_between(iterations, train_mean - train_std, train_mean + train_std,
                    color="#3498DB", alpha=0.2)
    ax.plot(iterations, val_mean,   label="Validation Loss", color="#E74C3C", linewidth=2)
    ax.fill_between(iterations, val_mean - val_std, val_mean + val_std,
                    color="#E74C3C", alpha=0.2)

    best_iter = int(np.argmin(val_mean))
    ax.axvline(x=best_iter + 1, color="green", linestyle="--", alpha=0.7, linewidth=2)
    ax.scatter([best_iter + 1], [val_mean[best_iter]], color="green", s=100, zorder=5,
               label=f"Mejor iter: {best_iter + 1}  (loss={val_mean[best_iter]:.4f})")

    ax.set_xlabel("Iterations (n_estimators)", fontsize=12)
    ax.set_ylabel("Binary Logloss", fontsize=12)
    sufijo = "Optimizado" if tag_optimizado else "Sin Optimizar"
    ax.set_title(
        f"Curva de Pérdida - CatBoost {fase}\n"
        f"(Media ± Std de {cv_folds}-Fold CV) - {sufijo}",
        fontsize=14,
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    tag_archivo = "opt" if tag_optimizado else "porDefecto"
    fig_path = output_dir / f"curva_perdida_catboost_{tag_archivo}_{fase}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Figura guardada: {fig_path}")

    gap = train_mean[-1] - val_mean[-1]
    print(f"   Mejor iter: {best_iter + 1}  |  "
          f"Train Loss final: {train_mean[-1]:.4f}  |  "
          f"Val Loss mínimo: {val_mean[best_iter]:.4f}  |  "
          f"Gap: {gap:.4f}")


def resumen_cv(cv_results: dict, fase: str, modelo: str) -> pd.DataFrame:
    """Genera un DataFrame de resumen con media y std de las métricas CV."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    summary = {"modelo": modelo, "fase": fase}
    for metric in metrics:
        summary[f"{metric}_val_mean"]   = cv_results[f"test_{metric}"].mean()
        summary[f"{metric}_val_std"]    = cv_results[f"test_{metric}"].std()
        summary[f"{metric}_train_mean"] = cv_results[f"train_{metric}"].mean()
        summary[f"{metric}_train_std"]  = cv_results[f"train_{metric}"].std()
    return pd.DataFrame([summary])


def _imprime_resumen_cv(cv_results: dict, cv_folds: int, fase: str) -> None:
    """Imprime tabla de métricas train/val por fold y su resumen."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    print(f"\n Resultados por fold:")
    for i in range(cv_folds):
        print(f"\n  Fold {i + 1}:")
        for m in metrics:
            print(f"    {m:<10} | Train: {cv_results[f'train_{m}'][i]:.4f} "
                  f"| Val: {cv_results[f'test_{m}'][i]:.4f}")

    print(f"\n Resumen Cross-Validation ({fase}):")
    print(f"   {'Métrica':<12} {'Train Mean':>12} {'Train Std':>12} {'Val Mean':>12} {'Val Std':>12}")
    print(f"   {'-' * 60}")
    for m in metrics:
        marker = " ****" if m == "f1" else ""
        print(
            f"   {m:<12} "
            f"{cv_results[f'train_{m}'].mean():>12.4f} "
            f"{cv_results[f'train_{m}'].std():>12.4f} "
            f"{cv_results[f'test_{m}'].mean():>12.4f} "
            f"{cv_results[f'test_{m}'].std():>12.4f}"
            f"{marker}"
        )


# ==============================================================================
# ENTRENAMIENTO CON PARÁMETROS POR DEFECTO
# ==============================================================================

def entrena_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    fase: str,
    cat_features_idx: list,
    cv_folds: int = 5,
    output_dir_figures: Path = OUTPUT_DIR_FIGURES,
) -> dict:

    mlflow.end_run()

    class_weights = _calcula_class_weights(y_train)

    print("=" * 80)
    print(f"  ENTRENAMIENTO CATBOOST - FASE {fase} (SIN OPTIMIZAR)")
    print("=" * 80)
    print(f"\n  Variables              : {X_train.shape[1]}")
    print(f"  Registros entrenamiento: {X_train.shape[0]}")
    print(f"  Categóricas nativas    : {len(cat_features_idx)}")
    print(f"  class_weights          : {{0: 1.0, 1: {class_weights[1]:.2f}}}")
    print(f"  Hiperparámetros por defecto:")
    print(f"    iterations=1000, depth=6, learning_rate=0.1, l2_leaf_reg=3, border_count=254")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {
        k: [] for k in [
            "train_accuracy", "test_accuracy",
            "train_precision", "test_precision",
            "train_recall", "test_recall",
            "train_f1", "test_f1",
            "train_roc_auc", "test_roc_auc",
        ]
    }
    train_losses, val_losses = [], []
    modelo_catb = None

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_idx]; y_fold_train = y_train.iloc[train_idx]
        X_fold_val   = X_train.iloc[val_idx];   y_fold_val   = y_train.iloc[val_idx]

        modelo_catb = CatBoostClassifier(
            iterations=1000,
            depth=6,
            learning_rate=0.1,
            l2_leaf_reg=3,
            border_count=254,
            class_weights=class_weights,
            loss_function="Logloss",
            eval_metric="F1",
            cat_features=cat_features_idx,
            random_seed=RANDOM_STATE,
            verbose=False,
        )
        # eval_set con fold de validación real → curvas train/val correctas
        modelo_catb.fit(
            X_fold_train, y_fold_train,
            eval_set=(X_fold_val, y_fold_val),
        )

        evals = modelo_catb.get_evals_result()
        train_losses.append(evals["learn"]["Logloss"])
        val_losses.append(evals["validation"]["Logloss"])

        for split_name, X_s, y_s in [("train", X_fold_train, y_fold_train),
                                      ("test",  X_fold_val,   y_fold_val)]:
            y_pred  = modelo_catb.predict(X_s)
            y_proba = modelo_catb.predict_proba(X_s)[:, 1]
            cv_results[f"{split_name}_accuracy"].append(accuracy_score(y_s, y_pred))
            cv_results[f"{split_name}_precision"].append(
                precision_score(y_s, y_pred, pos_label=1, zero_division=0))
            cv_results[f"{split_name}_recall"].append(
                recall_score(y_s, y_pred, pos_label=1, zero_division=0))
            cv_results[f"{split_name}_f1"].append(
                f1_score(y_s, y_pred, pos_label=1, zero_division=0))
            cv_results[f"{split_name}_roc_auc"].append(roc_auc_score(y_s, y_proba))

    for key in cv_results:
        cv_results[key] = np.array(cv_results[key])

    _imprime_resumen_cv(cv_results, cv_folds, fase)
    _grafica_curva_perdida(
        train_losses, val_losses, fase, cv_folds,
        tag_optimizado=False,
        output_dir=output_dir_figures,
    )

    # Registro en MLflow
    mlflow.set_experiment("TFM_Dropout_Prediction")
    with mlflow.start_run(run_name=f"CatBoost_CV5_{fase}"):
        mlflow.set_tag("modelo", "Params por default")
        mlflow.set_tag("tipo",   "Validacion cruzada")
        mlflow.log_params(modelo_catb.get_params())
        mlflow.log_param("cv_folds",   cv_folds)
        mlflow.log_param("n_features", X_train.shape[1])
        for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            mlflow.log_metric(f"test_{m}_mean", round(float(cv_results[f"test_{m}"].mean()), 4))
            mlflow.log_metric(f"test_{m}_std",  round(float(cv_results[f"test_{m}"].std()),  4))

    return {
        "phase":      fase,
        "model":      modelo_catb,
        "n_features": X_train.shape[1],
        "cv_results": cv_results,
    }


# ==============================================================================
# ENTRENAMIENTO CON OPTUNA
# ==============================================================================

def entrena_catBoost_con_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    fase: str,
    cat_features_idx: list,
    n_trials: int = 25,
    cv_folds: int = 5,
    output_dir_figures: Path = OUTPUT_DIR_FIGURES,
) -> dict:

    class_weights = _calcula_class_weights(y_train)

    print("=" * 80)
    print(f"  OPTIMIZACIÓN CATBOOST CON OPTUNA - FASE {fase}")
    print("=" * 80)
    print(f"\n  Variables              : {X_train.shape[1]}")
    print(f"  Registros entrenamiento: {X_train.shape[0]}")
    print(f"  Trials Optuna          : {n_trials}")
    print(f"  Categóricas nativas    : {len(cat_features_idx)}")
    print(f"  Métrica a optimizar    : F1-score (clase Dropout = 1)")
    print(f"  class_weights base     : {{0: 1.0, 1: {class_weights[1]:.2f}}}")

    # ------------------------------------------------------------------
    # Función objetivo
    # ------------------------------------------------------------------
    def objective(trial):
        params = {
            "iterations":          trial.suggest_int("iterations", 200, 1200),
            "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth":               trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "border_count":        trial.suggest_int("border_count", 128, 254),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
            "random_strength":     trial.suggest_float("random_strength", 0.0, 10.0),
            "auto_class_weights":  trial.suggest_categorical(
                "auto_class_weights", ["Balanced", "SqrtBalanced", None]),
            "cat_features":        cat_features_idx,
            "loss_function":       "Logloss",
            "eval_metric":         "F1",
            "random_seed":         RANDOM_STATE,
            "verbose":             False,
            "early_stopping_rounds": 50,
        }

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for tr_idx, val_idx in cv.split(X_train, y_train):
            try:
                m = CatBoostClassifier(**params)
                m.fit(
                    X_train.iloc[tr_idx], y_train.iloc[tr_idx],
                    eval_set=(X_train.iloc[val_idx], y_train.iloc[val_idx]),
                    verbose=False,
                )
                scores.append(
                    f1_score(y_train.iloc[val_idx], m.predict(X_train.iloc[val_idx]),
                             pos_label=1, zero_division=0)
                )
            except Exception:
                return 0.0
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    study = optuna.create_study(
        direction="maximize",
        study_name=f"F1-socre_{fase}",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params   = study.best_params
    best_f1_score = study.best_value

    print(f"\n{'=' * 70}")
    print(f"  MEJORES HIPERPARÁMETROS  —  F1-CV: {best_f1_score:.4f}")
    print(f"{'=' * 70}")
    for k, v in best_params.items():
        print(f"    {k}: {v}")

    # ------------------------------------------------------------------
    # CV final con mejores parámetros
    # ------------------------------------------------------------------
    final_params = {
        "iterations":          best_params["iterations"],
        "learning_rate":       best_params["learning_rate"],
        "depth":               best_params["depth"],
        "l2_leaf_reg":         best_params["l2_leaf_reg"],
        "border_count":        best_params["border_count"],
        "bagging_temperature": best_params["bagging_temperature"],
        "random_strength":     best_params["random_strength"],
        "auto_class_weights":  best_params["auto_class_weights"],
        "cat_features":        cat_features_idx,
        "loss_function":       "Logloss",
        "eval_metric":         "F1",
        "random_seed":         RANDOM_STATE,
        "verbose":             False,
        "early_stopping_rounds": 50,
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {
        k: [] for k in [
            "train_accuracy", "test_accuracy",
            "train_precision", "test_precision",
            "train_recall", "test_recall",
            "train_f1", "test_f1",
            "train_roc_auc", "test_roc_auc",
        ]
    }
    train_losses, val_losses = [], []
    modelo_catb_opt = None

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_idx]; y_fold_train = y_train.iloc[train_idx]
        X_fold_val   = X_train.iloc[val_idx];   y_fold_val   = y_train.iloc[val_idx]

        modelo_catb_opt = CatBoostClassifier(**final_params)
        modelo_catb_opt.fit(
            X_fold_train, y_fold_train,
            eval_set=(X_fold_val, y_fold_val),
        )

        evals = modelo_catb_opt.get_evals_result()
        train_losses.append(evals["learn"]["Logloss"])
        val_losses.append(evals["validation"]["Logloss"])

        for split_name, X_s, y_s in [("train", X_fold_train, y_fold_train),
                                      ("test",  X_fold_val,   y_fold_val)]:
            y_pred  = modelo_catb_opt.predict(X_s)
            y_proba = modelo_catb_opt.predict_proba(X_s)[:, 1]
            cv_results[f"{split_name}_accuracy"].append(accuracy_score(y_s, y_pred))
            cv_results[f"{split_name}_precision"].append(
                precision_score(y_s, y_pred, pos_label=1, zero_division=0))
            cv_results[f"{split_name}_recall"].append(
                recall_score(y_s, y_pred, pos_label=1, zero_division=0))
            cv_results[f"{split_name}_f1"].append(
                f1_score(y_s, y_pred, pos_label=1, zero_division=0))
            cv_results[f"{split_name}_roc_auc"].append(roc_auc_score(y_s, y_proba))

    for key in cv_results:
        cv_results[key] = np.array(cv_results[key])

    print(f"\n{'=' * 70}")
    print(f"  RESUMEN CROSS-VALIDATION (Optimizado) — FASE {fase}")
    print(f"{'=' * 70}")
    _imprime_resumen_cv(cv_results, cv_folds, fase)

    _grafica_curva_perdida(
        train_losses, val_losses, fase, cv_folds,
        tag_optimizado=True,
        output_dir=output_dir_figures,
    )

    # Registro en MLflow
    with mlflow.start_run(run_name=f"OptunaCoste_CatBoost_CV5_{fase}"):
        mlflow.set_tag("modelo", "Baseline - Optimizado_Optuna")
        mlflow.set_tag("tipo",   "Validacion cruzada")
        mlflow.log_params(modelo_catb_opt.get_params())
        mlflow.log_param("n_trials",   n_trials)
        mlflow.log_param("cv_folds",   cv_folds)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_metric("optuna_best_f1_cv", best_f1_score)

        for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            mlflow.log_metric(f"test_{m}_mean", round(float(cv_results[f"test_{m}"].mean()), 4))
            mlflow.log_metric(f"test_{m}_std",  round(float(cv_results[f"test_{m}"].std()),  4))

    return {
        "fase":            fase,
        "model":           modelo_catb_opt,
        "best_params":     best_params,
        "best_f1_sore_cv": best_f1_score,
        "cv_results":      cv_results,
        "study":           study,
    }


# ==============================================================================
# FUNCIÓN PRINCIPAL DEL PIPELINE
# ==============================================================================

def modelado_CatBoost(
    input_path: str | None = None,
    output_dir_figures: str | None = None,
    output_dir_models: str | None = None,
    mlruns_dir: str | None = None,
    n_trials: int = 25,
    cv_folds: int = 5,
    verbose: bool = True,
) -> None:

    # ------------------------------------------------------------------
    # Resolución de rutas
    # ------------------------------------------------------------------
    data_path  = Path(input_path)         if input_path         else DATA_PROCESSED_PATH
    fig_dir    = Path(output_dir_figures) if output_dir_figures else OUTPUT_DIR_FIGURES
    models_dir = Path(output_dir_models)  if output_dir_models  else OUTPUT_DIR_MODELS

    _mlruns_path = Path(mlruns_dir).resolve() if mlruns_dir else MLRUNS_DIR.resolve()
    mlruns_uri   = _mlruns_path.as_uri()

    fig_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_MODELS_GLOBAL.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Configuración MLflow
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(mlruns_uri)
    mlflow.set_experiment("TFM_Dropout_Prediction")

    experiment = mlflow.get_experiment_by_name("TFM_Dropout_Prediction")
    if verbose:
        print("=" * 80)
        print("  CONFIGURACIÓN MLFLOW")
        print("=" * 80)
        print(f"  Tracking URI  : {mlruns_uri}")
        print(f"  Experiment ID : {experiment.experiment_id if experiment else 'Nuevo'}")
        print(f"\n  Para visualizar resultados:\n    mlflow ui --port 5000")

    # ------------------------------------------------------------------
    # 1. Carga de datos
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 80)
        print("  1. CARGA DE DATOS PREPROCESADOS")
        print("=" * 80)

    df = pd.read_csv(data_path)
    if verbose:
        print(f"\n  Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
        print(f"\n  Target binario:")
        print(df[TARGET].value_counts().to_string())
        ratio = df[TARGET].value_counts()[0] / df[TARGET].value_counts()[1]
        print(f"\n  Ratio de desbalance: {ratio:.2f}:1")

    # ------------------------------------------------------------------
    # 2. Variables por fase temporal
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 80)
        print("  2. VARIABLES POR FASE TEMPORAL")
        print("=" * 80)
        print(f"  T0 (Matrícula)   : {len(VARS_T0)} variables")
        print(f"  T1 (Fin 1er Sem) : {len(VARS_T1)} variables (+{len(VARS_T1) - len(VARS_T0)})")
        print(f"  T2 (Fin 2do Sem) : {len(VARS_T2)} variables (+{len(VARS_T2) - len(VARS_T1)})")

    # ------------------------------------------------------------------
    # 3. Split estratificado
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 80)
        print("  3. SPLIT TRAIN / TEST  (80 / 20 estratificado)")
        print("=" * 80)

    X = df[VARS_T2].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    if verbose:
        print(f"\n  Train : {X_train.shape[0]} registros ({X_train.shape[0]/len(df)*100:.1f}%)")
        print(f"  Test  : {X_test.shape[0]}  registros ({X_test.shape[0]/len(df)*100:.1f}%)")
        ratio_tr = y_train.value_counts()[0] / y_train.value_counts()[1]
        ratio_te = y_test.value_counts()[0]  / y_test.value_counts()[1]
        print(f"\n  Ratio Train : {ratio_tr:.2f}:1  |  Ratio Test: {ratio_te:.2f}:1")

    # ------------------------------------------------------------------
    # 4-7. Loop por fase
    # ------------------------------------------------------------------
    csv_path_cb = models_dir / "cv_summary_CatBoost.csv"
    df_acumulado = pd.DataFrame()

    for fase in ["T0", "T1", "T2"]:
        if verbose:
            print("\n" + "=" * 80)
            print(f"  FASE {fase}")
            print("=" * 80)

        X_tr, X_te, features, prep = preprocesamiento_catboost(X_train, X_test, y_train, fase)
        cat_idx = prep["cat_features_idx"]

        if verbose:
            print(f"\n  Dimensiones post-preprocesamiento:")
            print(f"    Train : {X_tr.shape}  |  Test: {X_te.shape}  |  Features: {len(features)}")
            print(f"    Categóricas nativas: {prep['cat_features_names']}")

        # --- Sin optimización ---
        results_base = entrena_catboost(X_tr, y_train, fase, cat_idx, cv_folds, fig_dir)

        df_base = resumen_cv(results_base["cv_results"], fase, "CatBoost")
        df_acumulado = pd.concat([df_acumulado, df_base], ignore_index=True)
        df_acumulado.to_csv(csv_path_cb, index=False)

        # --- Optuna ---
        results_opt = entrena_catBoost_con_optuna(
            X_tr, y_train, fase, cat_idx, n_trials, cv_folds, fig_dir
        )

        if verbose:
            print(f"\n  Comparación F1-score — {fase}:")
            print(f"    Sin optimización : {results_base['cv_results']['test_f1'].mean():.4f}")
            print(f"    Optuna           : {results_opt['best_f1_sore_cv']:.4f}")

        df_opt = resumen_cv(results_opt["cv_results"], fase, "CatBoost_opt")
        df_acumulado = pd.concat([df_acumulado, df_opt], ignore_index=True)
        df_acumulado.to_csv(csv_path_cb, index=False)

    # ------------------------------------------------------------------
    # 8. Resumen final CatBoost
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  RESUMEN FINAL — CATBOOST (CROSS-VALIDATION)")
    print("=" * 80)
    df_cb_final = pd.read_csv(csv_path_cb)
    print(df_cb_final.to_string(index=False))
    print(f"\n  Resultados guardados en: {csv_path_cb}")

    # ------------------------------------------------------------------
    # 9. Resumen comparativo global acumulado (RL + RF + XGB + LGB + CatBoost)
    # ------------------------------------------------------------------
    csv_path_global = OUTPUT_DIR_MODELS_GLOBAL / "cv_summary_entrenamiento.csv"

    if csv_path_global.exists():
        df_previo = pd.read_csv(csv_path_global)
        df_global = pd.concat([df_previo, df_cb_final], ignore_index=True)
    else:
        df_global = df_cb_final.copy()
        if verbose:
            print(f"\n  Aviso: no se encontró {csv_path_global}. "
                  f"Se crea con los resultados de CatBoost únicamente.")

    df_global.to_csv(csv_path_global, index=False)
    print(f"\n  Resumen comparativo global guardado en: {csv_path_global}")



# Funcion principal
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de entrenamiento CatBoost"
    )
    parser.add_argument("--input",    "-i", type=str, default=None,
                        help="Ruta al CSV preprocesado")
    parser.add_argument("--figures",  "-f", type=str, default=None,
                        help="Directorio de salida para figuras")
    parser.add_argument("--models",   "-m", type=str, default=None,
                        help="Directorio de salida para reportes CSV")
    parser.add_argument("--mlruns",   "-r", type=str, default=None,
                        help="URI de tracking MLflow")
    parser.add_argument("--n-trials", "-t", type=int, default=25,
                        help="Número de trials para Optuna por fase (default: 25)")
    parser.add_argument("--cv-folds", "-k", type=int, default=5,
                        help="Número de folds para Cross-Validation (default: 5)")
    parser.add_argument("--quiet",    "-q", action="store_true",
                        help="Ejecutar sin mensajes de progreso")

    args = parser.parse_args()

    modelado_CatBoost(
        input_path=args.input,
        output_dir_figures=args.figures,
        output_dir_models=args.models,
        mlruns_dir=args.mlruns,
        n_trials=args.n_trials,
        cv_folds=args.cv_folds,
        verbose=not args.quiet,
    )

    print("\n" + "=" * 80)
    print("  MODELADO CATBOOST COMPLETADO")
    print("=" * 80)


if __name__ == "__main__":
    main()
