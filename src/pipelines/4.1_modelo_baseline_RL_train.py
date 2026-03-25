#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================================================
# MODELADO BASELINE - REGRESIÓN LOGÍSTICA (RL)
# ==============================================================================


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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline  # noqa: F401  (disponible para extensiones)

# Modelo
from sklearn.linear_model import LogisticRegression

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

DATA_PROCESSED_PATH  = PROJECT_ROOT / "data" / "processed" / "preprocessed_data.csv"
OUTPUT_DIR_FIGURES   = PROJECT_ROOT / "outputs" / "figures"  / "modelado" / "baseline_RL"
OUTPUT_DIR_MODELS    = PROJECT_ROOT / "outputs" / "models"   / "baseline_RL"
MLRUNS_DIR           = PROJECT_ROOT / "mlruns"

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
            "binarias":        VARS_BINARIAS_T0,
            "numericas":       VARS_NUMERICAS_T0 + VARS_ORDINALES_T0,
            "categoricas_ohe": VARS_CATEGORICAS_AGRUPADAS_T0,
            "categoricas_te":  VARS_TARGET_ENCODING_T0,
            "todas":           VARS_T0,
        }
    elif fase == "T1":
        return {
            "binarias":        VARS_BINARIAS_T0 + VARS_BINARIAS_T1,
            "numericas":       VARS_NUMERICAS_T0 + VARS_ORDINALES_T0 + VARS_NUMERICAS_T1,
            "categoricas_ohe": VARS_CATEGORICAS_AGRUPADAS_T0,
            "categoricas_te":  VARS_TARGET_ENCODING_T0,
            "todas":           VARS_T1,
        }
    elif fase == "T2":
        return {
            "binarias":        VARS_BINARIAS_T0 + VARS_BINARIAS_T1,
            "numericas":       VARS_NUMERICAS_T0 + VARS_ORDINALES_T0 + VARS_NUMERICAS_T1 + VARS_NUMERICAS_T2,
            "categoricas_ohe": VARS_CATEGORICAS_AGRUPADAS_T0,
            "categoricas_te":  VARS_TARGET_ENCODING_T0,
            "todas":           VARS_T2,
        }
    else:
        raise ValueError(f"Fase no válida: {fase}. Usar 'T0', 'T1' o 'T2'.")


def preprocesamiento_RL(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    fase: str,
) -> tuple:

    variables_fase = obtiene_variables_por_fase(fase)

    X_train_fase = X_train[variables_fase["todas"]].copy()
    X_test_fase  = X_test[variables_fase["todas"]].copy()

    # ------------------------------------------------------------------
    # Variables zero-inflated → transformación log1p
    # ------------------------------------------------------------------
    vars_zero_inflated = [
        "curricular_units_1st_sem_credited",
        "curricular_units_2nd_sem_credited",
        "curricular_units_1st_sem_without_evaluations",
        "curricular_units_2nd_sem_without_evaluations",
    ]
    vars_zi_fase = [v for v in vars_zero_inflated if v in X_train_fase.columns]

    for col in vars_zi_fase:
        X_train_fase[col] = np.log1p(X_train_fase[col])
        X_test_fase[col]  = np.log1p(X_test_fase[col])

    X_train_fase["age_at_enrollment"] = np.log1p(X_train_fase["age_at_enrollment"])
    X_test_fase["age_at_enrollment"]  = np.log1p(X_test_fase["age_at_enrollment"])

    # ------------------------------------------------------------------
    # Target Encoding para 'course'
    # ------------------------------------------------------------------
    te = TargetEncoder(cols=variables_fase["categoricas_te"], smoothing=0.3)

    for col in variables_fase["categoricas_te"]:
        X_train_fase[col + "_encoded"] = te.fit_transform(X_train_fase[[col]], y_train)[col]
        X_test_fase[col + "_encoded"]  = te.transform(X_test_fase[[col]])[col]
        X_train_fase = X_train_fase.drop(columns=[col])
        X_test_fase  = X_test_fase.drop(columns=[col])

    vars_numericas_updated = (
        variables_fase["numericas"]
        + [col + "_encoded" for col in variables_fase["categoricas_te"]]
    )

    # ------------------------------------------------------------------
    # One-Hot Encoding para categóricas agrupadas
    # ------------------------------------------------------------------
    X_train_fase = pd.get_dummies(
        X_train_fase,
        columns=variables_fase["categoricas_ohe"],
        drop_first=True,
        dtype=int,
    )
    X_test_fase = pd.get_dummies(
        X_test_fase,
        columns=variables_fase["categoricas_ohe"],
        drop_first=True,
        dtype=int,
    )

    # Alinear columnas train/test
    X_train_fase, X_test_fase = X_train_fase.align(
        X_test_fase, join="left", axis=1, fill_value=0
    )

    # ------------------------------------------------------------------
    # StandardScaler para numéricas
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    cols_to_scale = [c for c in vars_numericas_updated if c in X_train_fase.columns]

    X_train_fase[cols_to_scale] = scaler.fit_transform(X_train_fase[cols_to_scale])
    X_test_fase[cols_to_scale]  = scaler.transform(X_test_fase[cols_to_scale])

    feature_names  = X_train_fase.columns.tolist()
    preprocessors  = {
        "target_encoder": te,
        "scaler":         scaler,
        "feature_names":  feature_names,
    }

    return X_train_fase, X_test_fase, feature_names, preprocessors


# ==============================================================================
# FUNCIONES DE ENTRENAMIENTO
# ==============================================================================

def _ejecuta_cv(modelo_params: dict, X_train: pd.DataFrame, y_train: pd.Series,
                cv_folds: int) -> dict:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {
        k: []
        for k in [
            "train_accuracy", "test_accuracy",
            "train_precision", "test_precision",
            "train_recall", "test_recall",
            "train_f1", "test_f1",
            "train_roc_auc", "test_roc_auc",
        ]
    }

    modelo = None
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold_tr = X_train.iloc[train_idx];  y_fold_tr = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx];   y_fold_val = y_train.iloc[val_idx]

        modelo = LogisticRegression(**modelo_params)
        modelo.fit(X_fold_tr, y_fold_tr)

        for split_name, X_s, y_s in [("train", X_fold_tr, y_fold_tr), ("test", X_fold_val, y_fold_val)]:
            y_pred  = modelo.predict(X_s)
            y_proba = modelo.predict_proba(X_s)[:, 1]
            cv_results[f"{split_name}_accuracy"].append(accuracy_score(y_s, y_pred))
            cv_results[f"{split_name}_precision"].append(precision_score(y_s, y_pred, pos_label=1, zero_division=0))
            cv_results[f"{split_name}_recall"].append(recall_score(y_s, y_pred, pos_label=1, zero_division=0))
            cv_results[f"{split_name}_f1"].append(f1_score(y_s, y_pred, pos_label=1, zero_division=0))
            cv_results[f"{split_name}_roc_auc"].append(roc_auc_score(y_s, y_proba))

    for key in cv_results:
        cv_results[key] = np.array(cv_results[key])

    return cv_results, modelo


def _grafica_curva_regularizacion(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    fase: str,
    cv_folds: int,
    extra_params: dict,
    tag_optimizado: bool,
    best_C_optuna: float | None = None,
    output_dir: Path = OUTPUT_DIR_FIGURES,
) -> None:

    C_range = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    all_train_f1, all_val_f1 = [], []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_ftr = X_train.iloc[train_idx]; y_ftr = y_train.iloc[train_idx]
        X_fv  = X_train.iloc[val_idx];  y_fv  = y_train.iloc[val_idx]

        train_row, val_row = [], []
        for C in C_range:
            params = {"penalty": "l2", "C": C, "solver": "saga",
                      "class_weight": "balanced", "random_state": RANDOM_STATE, **extra_params}
            m = LogisticRegression(**params)
            m.fit(X_ftr, y_ftr)
            train_row.append(f1_score(y_ftr, m.predict(X_ftr), pos_label=1, zero_division=0))
            val_row.append(f1_score(y_fv,  m.predict(X_fv),  pos_label=1, zero_division=0))

        all_train_f1.append(train_row)
        all_val_f1.append(val_row)

    train_mean = np.array(all_train_f1).mean(axis=0)
    train_std  = np.array(all_train_f1).std(axis=0)
    val_mean   = np.array(all_val_f1).mean(axis=0)
    val_std    = np.array(all_val_f1).std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(C_range, train_mean, label="Train F1",      color="#3498DB", linewidth=2, marker="o")
    ax.fill_between(C_range, train_mean - train_std, train_mean + train_std, color="#3498DB", alpha=0.2)
    ax.plot(C_range, val_mean,   label="Validation F1", color="#E74C3C", linewidth=2, marker="s")
    ax.fill_between(C_range, val_mean - val_std, val_mean + val_std, color="#E74C3C", alpha=0.2)

    ax.set_xscale("log")
    ax.set_xlabel("C (Regularización Inversa: menor C = más regularización)", fontsize=12)
    ax.set_ylabel("F1-Score", fontsize=12)
    sufijo = "Optimizado" if tag_optimizado else "Sin Optimizar"
    ax.set_title(
        f"Curva de Regularización - Regresión Logística {fase}\n"
        f"(Media ± Std de {cv_folds}-Fold CV) - {sufijo}",
        fontsize=14,
    )

    best_idx    = int(np.argmax(val_mean))
    best_C_curv = C_range[best_idx]
    ax.axvline(x=best_C_curv, color="green", linestyle="--", alpha=0.7, linewidth=2)
    ax.scatter([best_C_curv], [val_mean[best_idx]], color="green", s=150, zorder=5,
               label=f"Mejor C (curva): {best_C_curv}")

    if not tag_optimizado:
        idx_def = C_range.index(1.0)
        ax.axvline(x=1.0, color="purple", linestyle=":", alpha=0.7, linewidth=2)
        ax.scatter([1.0], [val_mean[idx_def]], color="purple", s=150, zorder=5,
                   label=f"C default: 1.0 → F1={val_mean[idx_def]:.4f}")

    if best_C_optuna is not None and best_C_optuna != best_C_curv:
        ax.axvline(x=best_C_optuna, color="orange", linestyle=":", alpha=0.7, linewidth=2)
        ax.scatter([best_C_optuna], [np.interp(best_C_optuna, C_range, val_mean)],
                   color="orange", s=150, zorder=5, label=f"C Optuna: {best_C_optuna:.4f}")

    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    tag_archivo = "opt" if tag_optimizado else "baseline"
    fig_path = output_dir / f"curva_regularizacion_rl_{tag_archivo}_{fase}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Figura guardada: {fig_path}")


def resumen_cv(cv_results: dict, fase: str, modelo: str) -> pd.DataFrame:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    summary = {"modelo": modelo, "fase": fase}

    for metric in metrics:
        summary[f"{metric}_val_mean"]   = cv_results[f"test_{metric}"].mean()
        summary[f"{metric}_val_std"]    = cv_results[f"test_{metric}"].std()
        summary[f"{metric}_train_mean"] = cv_results[f"train_{metric}"].mean()
        summary[f"{metric}_train_std"]  = cv_results[f"train_{metric}"].std()

    return pd.DataFrame([summary])


def _imprime_resumen_cv(cv_results: dict, cv_folds: int, fase: str) -> None:
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
# ENTRENAMIENTO BASELINE (parámetros por defecto)
# ==============================================================================

def entrena_RL(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    fase: str,
    cv_folds: int = 5,
    output_dir_figures: Path = OUTPUT_DIR_FIGURES,
) -> dict:
    mlflow.end_run()

    print("===========================================================================================")
    print(f"  ENTRENAMIENTO REGRESIÓN LOGÍSTICA - FASE {fase} (SIN OPTIMIZAR)")
    print("===========================================================================================")
    print(f"\n  Variables              : {X_train.shape[1]}")
    print(f"  Registros entrenamiento: {X_train.shape[0]}")
    print(f"  Cross-Validation       : {cv_folds}-fold estratificado")

    modelo_params = {
        "penalty":      "l2",
        "C":            1.0,
        "solver":       "lbfgs",
        "class_weight": "balanced",
        "max_iter":     100,
        "random_state": RANDOM_STATE,
    }

    cv_results, modelo = _ejecuta_cv(modelo_params, X_train, y_train, cv_folds)
    _imprime_resumen_cv(cv_results, cv_folds, fase)

    # Curva de regularización
    _grafica_curva_regularizacion(
        X_train, y_train, fase, cv_folds,
        extra_params={"max_iter": 1000},
        tag_optimizado=False,
        output_dir=output_dir_figures,
    )

    # Registro en MLflow
    mlflow.set_experiment("TFM_Dropout_Prediction")
    with mlflow.start_run(run_name=f"NoOpt_RegresionLogistica_CV5_{fase}"):
        mlflow.set_tag("modelo", "Baseline - Params por default")
        mlflow.set_tag("tipo", "Validación cruzada")
        mlflow.log_params(modelo.get_params())
        mlflow.log_param("cv_folds",    cv_folds)
        mlflow.log_param("n_features",  X_train.shape[1])
        for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            mlflow.log_metric(f"test_{m}_mean", round(float(cv_results[f"test_{m}"].mean()), 4))
            mlflow.log_metric(f"test_{m}_std",  round(float(cv_results[f"test_{m}"].std()),  4))

    return {
        "fase":       fase,
        "modelo":     modelo,
        "n_features": X_train.shape[1],
        "cv_results": cv_results,
    }


# ==============================================================================
# ENTRENAMIENTO CON OPTUNA
# ==============================================================================

def entrena_RL_con_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    fase: str,
    n_trials: int = 25,
    cv_folds: int = 5,
    output_dir_figures: Path = OUTPUT_DIR_FIGURES,
) -> dict:

    print("===========================================================================================")
    print(f"  OPTIMIZACIÓN REGRESIÓN LOGÍSTICA CON OPTUNA - FASE {fase}")
    print("===========================================================================================")
    print(f"\n  Variables              : {X_train.shape[1]}")
    print(f"  Registros entrenamiento: {X_train.shape[0]}")
    print(f"  Trials Optuna          : {n_trials}")
    print(f"  Métrica a optimizar    : F1-score (clase Dropout = 1)")

    # ------------------------------------------------------------------
    # Función objetivo
    # ------------------------------------------------------------------
    def objective(trial):
        params = {
            "penalty":      "l2",
            "C":            trial.suggest_float("C", 0.001, 10.0, log=True),
            "solver":       "saga",
            "class_weight": "balanced",
            "max_iter":     trial.suggest_int("max_iter", 500, 2000),
            "random_state": RANDOM_STATE,
        }
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for tr_idx, val_idx in cv.split(X_train, y_train):
            try:
                m = LogisticRegression(**params)
                m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
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
        study_name=f"F1-score_{fase}",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params    = study.best_params
    best_f1_cv     = study.best_value

    print(f"\n{'===========================================================================================' }")
    print(f"  MEJORES HIPERPARÁMETROS  —  F1-CV: {best_f1_cv:.4f}")
    print(f"{'==========================================================================================='}")
    for k, v in best_params.items():
        print(f"    {k}: {v}")

    # ------------------------------------------------------------------
    # CV final con mejores parámetros
    # ------------------------------------------------------------------
    final_params = {
        "penalty":      "l2",
        "C":            best_params["C"],
        "solver":       "saga",
        "class_weight": "balanced",
        "max_iter":     best_params["max_iter"],
        "random_state": RANDOM_STATE,
    }

    cv_results, modelo_final = _ejecuta_cv(final_params, X_train, y_train, cv_folds)

    print(f"\n{'==========================================================================================='}")
    print(f"  RESUMEN CROSS-VALIDATION (Optimizado) — FASE {fase}")
    print(f"{'==========================================================================================='}")
    _imprime_resumen_cv(cv_results, cv_folds, fase)

    # Curva de regularización post-optimización
    _grafica_curva_regularizacion(
        X_train, y_train, fase, cv_folds,
        extra_params={"max_iter": best_params["max_iter"]},
        tag_optimizado=True,
        best_C_optuna=best_params["C"],
        output_dir=output_dir_figures,
    )

    # Registro en MLflow
    with mlflow.start_run(run_name=f"Optuna_RegresionLogistica_CV5_{fase}"):
        mlflow.set_tag("modelo", "Baseline - Optimizado_Optuna")
        mlflow.set_tag("tipo",   "Validacion cruzada")
        mlflow.log_params(modelo_final.get_params())
        mlflow.log_param("n_trials",   n_trials)
        mlflow.log_param("cv_folds",   cv_folds)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_metric("optuna_best_f1_cv", round(best_f1_cv, 4))
        for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            mlflow.log_metric(f"test_{m}_mean", round(float(cv_results[f"test_{m}"].mean()), 4))
            mlflow.log_metric(f"test_{m}_std",  round(float(cv_results[f"test_{m}"].std()),  4))

    return {
        "fase":           fase,
        "model":          modelo_final,
        "best_params":    best_params,
        "best_f1_sore_cv": best_f1_cv,
        "cv_results":     cv_results,
        "study":          study,
    }


# ==============================================================================
# FUNCIÓN PRINCIPAL DEL PIPELINE
# ==============================================================================

def modelado_baseline_RL(
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
    data_path   = Path(input_path)         if input_path         else DATA_PROCESSED_PATH
    fig_dir     = Path(output_dir_figures) if output_dir_figures else OUTPUT_DIR_FIGURES
    models_dir  = Path(output_dir_models)  if output_dir_models  else OUTPUT_DIR_MODELS
    _mlruns_path = Path(mlruns_dir).resolve() if mlruns_dir else MLRUNS_DIR.resolve()
    mlruns_uri   = _mlruns_path.as_uri()   # → file:///C:/... en Windows, file:///home/... en Linux

    fig_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Configuración MLflow
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(mlruns_uri)
    mlflow.set_experiment("TFM_Dropout_Prediction")

    experiment = mlflow.get_experiment_by_name("TFM_Dropout_Prediction")
    if verbose:
        print("===========================================================================================")
        print("  CONFIGURACIÓN MLFLOW")
        print("===========================================================================================")
        print(f"  Tracking URI   : {mlruns_uri}")
        print(f"  Experiment ID  : {experiment.experiment_id if experiment else 'Nuevo'}")
        print(f"\n  Para visualizar resultados:\n    mlflow ui --port 5000")

    # ------------------------------------------------------------------
    # 1. Carga de datos
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "===========================================================================================")
        print("  1. CARGA DE DATOS PREPROCESADOS")
        print("===========================================================================================")

    df = pd.read_csv(data_path)
    if verbose:
        print(f"\n  Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
        print(f"\n  Target binario:")
        print(df[TARGET].value_counts().to_string())
        ratio = df[TARGET].value_counts()[0] / df[TARGET].value_counts()[1]
        print(f"\n  Ratio: {ratio:.2f}:1")

    # ------------------------------------------------------------------
    # 2. Variables por fase temporal
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "===========================================================================================")
        print("  2. VARIABLES POR FASE TEMPORAL")
        print("===========================================================================================")
        print(f"  T0 (Matrícula)      : {len(VARS_T0)} variables")
        print(f"  T1 (Fin 1er Sem)    : {len(VARS_T1)} variables (+{len(VARS_T1) - len(VARS_T0)})")
        print(f"  T2 (Fin 2do Sem)    : {len(VARS_T2)} variables (+{len(VARS_T2) - len(VARS_T1)})")

    # ------------------------------------------------------------------
    # 3. Split estratificado
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "===========================================================================================")
        print("  3. SPLIT TRAIN / TEST  (80 / 20 estratificado)")
        print("===========================================================================================")

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
    # 4-7. Loop por fase: preprocesamiento → baseline → optuna → guardado
    # ------------------------------------------------------------------
    csv_path = models_dir / "cv_summary_RL.csv"
    df_acumulado = pd.DataFrame()

    for fase in ["T0", "T1", "T2"]:
        if verbose:
            print("\n" + "===========================================================================================")
            print(f"  FASE {fase}")
            print("===========================================================================================")

        # Preprocesamiento
        X_tr, X_te, features, prep = preprocesamiento_RL(X_train, X_test, y_train, fase)

        if verbose:
            print(f"\n  Dimensiones post-preprocesamiento:")
            print(f"    Train : {X_tr.shape}  |  Test: {X_te.shape}  |  Features: {len(features)}")

        # --- Baseline ---
        results_base = entrena_RL(X_tr, y_train, fase, cv_folds, fig_dir)

        df_base = resumen_cv(results_base["cv_results"], fase, "RegresionLogistica")
        df_acumulado = pd.concat([df_acumulado, df_base], ignore_index=True)
        df_acumulado.to_csv(csv_path, index=False)

        # --- Optuna ---
        results_opt = entrena_RL_con_optuna(X_tr, y_train, fase, n_trials, cv_folds, fig_dir)

        if verbose:
            print(f"\n  Comparación F1-score — {fase}:")
            print(f"    Baseline : {results_base['cv_results']['test_f1'].mean():.4f}")
            print(f"    Optuna   : {results_opt['best_f1_sore_cv']:.4f}")

        df_opt = resumen_cv(results_opt["cv_results"], fase, "RegresionLogistica_opt")
        df_acumulado = pd.concat([df_acumulado, df_opt], ignore_index=True)
        df_acumulado.to_csv(csv_path, index=False)

    # ------------------------------------------------------------------
    # 8. Resumen final
    # ------------------------------------------------------------------
    print("\n" + "===========================================================================================")
    print("  RESUMEN FINAL — REGRESIÓN LOGÍSTICA (CROSS-VALIDATION)")
    print("===========================================================================================")
    df_final = pd.read_csv(csv_path)
    print(df_final.to_string(index=False))
    print(f"\n  Resultados guardados en: {csv_path}")



# Funcion principal

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de entrenamiento Baseline - Regresión Logística"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Ruta al CSV preprocesado (default: data/processed/preprocessed_data.csv)",
    )
    parser.add_argument(
        "--figures", "-f",
        type=str,
        default=None,
        help="Directorio de salida para figuras (default: outputs/figures/modelado/baseline_RL)",
    )
    parser.add_argument(
        "--models", "-m",
        type=str,
        default=None,
        help="Directorio de salida para reportes CSV (default: outputs/models/baseline_RL)",
    )
    parser.add_argument(
        "--mlruns", "-r",
        type=str,
        default=None,
        help="URI de tracking MLflow (default: mlruns/)",
    )
    parser.add_argument(
        "--n-trials", "-t",
        type=int,
        default=25,
        help="Número de trials para Optuna por fase (default: 25)",
    )
    parser.add_argument(
        "--cv-folds", "-k",
        type=int,
        default=5,
        help="Número de folds para Cross-Validation (default: 5)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Ejecutar sin mensajes de progreso",
    )

    args = parser.parse_args()

    modelado_baseline_RL(
        input_path=args.input,
        output_dir_figures=args.figures,
        output_dir_models=args.models,
        mlruns_dir=args.mlruns,
        n_trials=args.n_trials,
        cv_folds=args.cv_folds,
        verbose=not args.quiet,
    )

    print("\n" + "===========================================================================================")
    print("  MODELADO BASELINE RL COMPLETADO")
    print("===========================================================================================")


if __name__ == "__main__":
    main()