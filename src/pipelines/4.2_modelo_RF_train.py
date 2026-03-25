#!/usr/bin/env python
# -*- coding: utf-8 -*-


# MODELADO RANDOM FOREST (RF)
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
from sklearn.preprocessing import LabelEncoder

# Modelo
from sklearn.ensemble import RandomForestClassifier

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

DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessed_data.csv"
OUTPUT_DIR_FIGURES  = PROJECT_ROOT / "outputs" / "figures" / "modelado" / "RF"
OUTPUT_DIR_MODELS   = PROJECT_ROOT / "outputs" / "models"  / "RF"
OUTPUT_DIR_MODELS_GLOBAL = PROJECT_ROOT / "outputs" / "models"
MLRUNS_DIR          = PROJECT_ROOT / "mlruns"

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
            "categoricas_le":  VARS_CATEGORICAS_AGRUPADAS_T0,
            "categoricas_te":  VARS_TARGET_ENCODING_T0,
            "all":             VARS_T0,
        }
    elif fase == "T1":
        return {
            "binarias":        VARS_BINARIAS_T0 + VARS_BINARIAS_T1,
            "numericas":       VARS_NUMERICAS_T0 + VARS_ORDINALES_T0 + VARS_NUMERICAS_T1,
            "categoricas_le":  VARS_CATEGORICAS_AGRUPADAS_T0,
            "categoricas_te":  VARS_TARGET_ENCODING_T0,
            "all":             VARS_T1,
        }
    elif fase == "T2":
        return {
            "binarias":        VARS_BINARIAS_T0 + VARS_BINARIAS_T1,
            "numericas":       VARS_NUMERICAS_T0 + VARS_ORDINALES_T0 + VARS_NUMERICAS_T1 + VARS_NUMERICAS_T2,
            "categoricas_le":  VARS_CATEGORICAS_AGRUPADAS_T0,
            "categoricas_te":  VARS_TARGET_ENCODING_T0,
            "all":             VARS_T2,
        }
    else:
        raise ValueError(f"Fase no válida: {fase}. Usar 'T0', 'T1' o 'T2'.")


def preprocesamiento_RF(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    fase: str,
) -> tuple:

    variables_fase = obtiene_variables_por_fase(fase)

    X_train_fase = X_train[variables_fase["all"]].copy()
    X_test_fase  = X_test[variables_fase["all"]].copy()

    label_encoders = {}

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
    # 2. Label Encoding para categóricas agrupadas
    # ------------------------------------------------------------------
    for col in variables_fase["categoricas_le"]:
        le = LabelEncoder()
        X_train_fase[col] = le.fit_transform(X_train_fase[col].astype(str))
        X_test_fase[col]  = le.transform(X_test_fase[col].astype(str))
        label_encoders[col] = le

    feature_names = X_train_fase.columns.tolist()
    preprocessors = {
        "target_encoder": te,
        "label_encoders": label_encoders,
        "feature_names":  feature_names,
    }

    return X_train_fase, X_test_fase, feature_names, preprocessors


# ==============================================================================
# FUNCIONES DE ENTRENAMIENTO
# ==============================================================================

def _ejecuta_cv_RF(modelo_params: dict, X_train: pd.DataFrame,
                   y_train: pd.Series, cv_folds: int) -> tuple:

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
    oob_scores = []
    modelo = None

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_ftr = X_train.iloc[train_idx]; y_ftr = y_train.iloc[train_idx]
        X_fv  = X_train.iloc[val_idx];  y_fv  = y_train.iloc[val_idx]

        modelo = RandomForestClassifier(**modelo_params)
        modelo.fit(X_ftr, y_ftr)

        if modelo_params.get("oob_score", False):
            oob_scores.append(modelo.oob_score_)

        for split_name, X_s, y_s in [("train", X_ftr, y_ftr), ("test", X_fv, y_fv)]:
            y_pred  = modelo.predict(X_s)
            y_proba = modelo.predict_proba(X_s)[:, 1]
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

    return cv_results, modelo, oob_scores


def _grafica_curva_aprendizaje_RF(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    fase: str,
    cv_folds: int,
    base_params: dict,
    tag_optimizado: bool,
    best_n_optuna: int | None = None,
    output_dir: Path = OUTPUT_DIR_FIGURES,
) -> None:

    n_estimators_range = [10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500]

    if tag_optimizado and best_n_optuna is not None:
        max_n = min(best_n_optuna + 100, 500)
        n_estimators_range = [n for n in n_estimators_range if n <= max_n]

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    all_train_f1, all_val_f1 = [], []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_ftr = X_train.iloc[train_idx]; y_ftr = y_train.iloc[train_idx]
        X_fv  = X_train.iloc[val_idx];  y_fv  = y_train.iloc[val_idx]

        train_row, val_row = [], []
        modelo_ws = RandomForestClassifier(
            **{k: v for k, v in base_params.items() if k != "n_estimators"},
            warm_start=True,
            n_estimators=0,
        )
        for n_est in n_estimators_range:
            modelo_ws.n_estimators = n_est
            modelo_ws.fit(X_ftr, y_ftr)
            train_row.append(f1_score(y_ftr, modelo_ws.predict(X_ftr), pos_label=1, zero_division=0))
            val_row.append(f1_score(y_fv,  modelo_ws.predict(X_fv),  pos_label=1, zero_division=0))

        all_train_f1.append(train_row)
        all_val_f1.append(val_row)

    train_mean = np.array(all_train_f1).mean(axis=0)
    train_std  = np.array(all_train_f1).std(axis=0)
    val_mean   = np.array(all_val_f1).mean(axis=0)
    val_std    = np.array(all_val_f1).std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_estimators_range, train_mean, label="Train F1",      color="#3498DB", linewidth=2, marker="o")
    ax.fill_between(n_estimators_range, train_mean - train_std, train_mean + train_std,
                    color="#3498DB", alpha=0.2)
    ax.plot(n_estimators_range, val_mean,   label="Validation F1", color="#E74C3C", linewidth=2, marker="s")
    ax.fill_between(n_estimators_range, val_mean - val_std, val_mean + val_std,
                    color="#E74C3C", alpha=0.2)

    ax.set_xlabel("Número de Árboles (n_estimators)", fontsize=12)
    ax.set_ylabel("F1-Score", fontsize=12)
    sufijo = "Optimizado" if tag_optimizado else "Sin Optimizar"
    ax.set_title(
        f"Curva de Aprendizaje - Random Forest {fase}\n"
        f"(Media ± Std de {cv_folds}-Fold CV) - {sufijo}",
        fontsize=14,
    )

    best_idx     = int(np.argmax(val_mean))
    best_n_curva = n_estimators_range[best_idx]
    ax.axvline(x=best_n_curva, color="green", linestyle="--", alpha=0.7, linewidth=2)
    ax.scatter([best_n_curva], [val_mean[best_idx]], color="green", s=150, zorder=5,
               label=f"Mejor n (curva): {best_n_curva}")

    if not tag_optimizado and 100 in n_estimators_range:
        idx_def = n_estimators_range.index(100)
        ax.axvline(x=100, color="purple", linestyle=":", alpha=0.7, linewidth=2)
        ax.scatter([100], [val_mean[idx_def]], color="purple", s=150, zorder=5,
                   label=f"n default: 100 → F1={val_mean[idx_def]:.4f}")

    if best_n_optuna is not None and best_n_optuna != best_n_curva:
        ax.axvline(x=best_n_optuna, color="orange", linestyle=":", alpha=0.7, linewidth=2)
        ax.scatter([best_n_optuna], [np.interp(best_n_optuna, n_estimators_range, val_mean)],
                   color="orange", s=150, zorder=5, label=f"n Optuna: {best_n_optuna}")

    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    tag_archivo = "opt" if tag_optimizado else "porDefecto"
    fig_path = output_dir / f"curva_aprendizaje_rf_{tag_archivo}_{fase}.png"
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

# Imprime tabla de métricas train/val por fold y su resumen
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
    print(f"   {'-------------------------------------------------------------------------------------'}")
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
# ENTRENAMIENTO parámetros por defecto
# ==============================================================================

def entrena_RF(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    fase: str,
    cv_folds: int = 5,
    output_dir_figures: Path = OUTPUT_DIR_FIGURES,
) -> dict:
    mlflow.end_run()

    print("==============================================================================")
    print(f"  ENTRENAMIENTO RANDOM FOREST - FASE {fase} (SIN OPTIMIZAR)")
    print("==============================================================================")
    print(f"\n  Variables              : {X_train.shape[1]}")
    print(f"  Registros entrenamiento: {X_train.shape[0]}")
    print(f"  Hiperparámetros por defecto:")
    print(f"    n_estimators=100, max_depth=None, min_samples_split=2,")
    print(f"    min_samples_leaf=1, max_features='sqrt', class_weight='balanced'")

    modelo_params = {
        "n_estimators":    100,
        "max_depth":       None,
        "min_samples_split": 2,
        "min_samples_leaf":  1,
        "max_features":    "sqrt",
        "bootstrap":       True,
        "class_weight":    "balanced",
        "random_state":    RANDOM_STATE,
        "n_jobs":          -1,
        "oob_score":       True,
    }

    cv_results, modelo, oob_scores = _ejecuta_cv_RF(modelo_params, X_train, y_train, cv_folds)
    _imprime_resumen_cv(cv_results, cv_folds, fase)

    if oob_scores:
        print(f"\n  OOB Score (último fold): {oob_scores[-1]:.4f}")

    # Curva de aprendizaje
    base_params_curva = {k: v for k, v in modelo_params.items()
                         if k not in ("n_estimators", "oob_score")}
    _grafica_curva_aprendizaje_RF(
        X_train, y_train, fase, cv_folds,
        base_params=base_params_curva,
        tag_optimizado=False,
        output_dir=output_dir_figures,
    )

    # Registro en MLflow
    mlflow.set_experiment("TFM_Dropout_Prediction")
    with mlflow.start_run(run_name=f"RandomForest_CV5_{fase}"):
        mlflow.set_tag("modelo", "Params por default")
        mlflow.set_tag("tipo",   "Validacion cruzada")
        mlflow.log_params(modelo.get_params())
        mlflow.log_param("cv_folds",   cv_folds)
        mlflow.log_param("n_features", X_train.shape[1])
        for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            mlflow.log_metric(f"test_{m}_mean", round(float(cv_results[f"test_{m}"].mean()), 4))
            mlflow.log_metric(f"test_{m}_std",  round(float(cv_results[f"test_{m}"].std()),  4))

    return {
        "phase":         fase,
        "model":         modelo,
        "n_features":    X_train.shape[1],
        "cv_results":    cv_results,
        "oob_score":     oob_scores[-1] if oob_scores else None,
        "oob_scores_cv": oob_scores,
    }


# ==============================================================================
# ENTRENAMIENTO CON OPTUNA
# ==============================================================================

def entrena_RF_con_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    fase: str,
    n_trials: int = 25,
    cv_folds: int = 5,
    output_dir_figures: Path = OUTPUT_DIR_FIGURES,
) -> dict:

    print("==============================================================================")
    print(f"  OPTIMIZACIÓN RANDOM FOREST CON OPTUNA - FASE {fase}")
    print("==============================================================================")
    print(f"\n  Variables              : {X_train.shape[1]}")
    print(f"  Registros entrenamiento: {X_train.shape[0]}")
    print(f"  Trials Optuna          : {n_trials}")
    print(f"  Métrica a optimizar    : F1-score (clase Dropout = 1)")

    # ------------------------------------------------------------------
    # Función objetivo
    # ------------------------------------------------------------------
    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "max_depth":         trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "class_weight":      trial.suggest_categorical(
                "class_weight", ["balanced", "balanced_subsample"]),
            "bootstrap":         True,
            "random_state":      RANDOM_STATE,
            "n_jobs":            -1,
        }
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for tr_idx, val_idx in cv.split(X_train, y_train):
            try:
                m = RandomForestClassifier(**params)
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
        study_name=f"F1-score_RF_{fase}",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_f1_cv  = study.best_value

    print(f"\n{'=============================================================================='}")
    print(f"  MEJORES HIPERPARÁMETROS  —  F1-CV: {best_f1_cv:.4f}")
    print(f"{'=============================================================================='}")
    for k, v in best_params.items():
        print(f"    {k}: {v}")

    # ------------------------------------------------------------------
    # Curva de aprendizaje post-optimización
    # ------------------------------------------------------------------
    base_params_curva = {
        "max_depth":         best_params["max_depth"],
        "min_samples_split": best_params["min_samples_split"],
        "min_samples_leaf":  best_params["min_samples_leaf"],
        "max_features":      best_params["max_features"],
        "class_weight":      best_params["class_weight"],
        "bootstrap":         True,
        "random_state":      RANDOM_STATE,
        "n_jobs":            -1,
    }
    _grafica_curva_aprendizaje_RF(
        X_train, y_train, fase, cv_folds,
        base_params=base_params_curva,
        tag_optimizado=True,
        best_n_optuna=best_params["n_estimators"],
        output_dir=output_dir_figures,
    )

    # ------------------------------------------------------------------
    # CV final con mejores parámetros (sin warm_start)
    # ------------------------------------------------------------------
    final_params = {
        "n_estimators":      best_params["n_estimators"],
        "max_depth":         best_params["max_depth"],
        "min_samples_split": best_params["min_samples_split"],
        "min_samples_leaf":  best_params["min_samples_leaf"],
        "max_features":      best_params["max_features"],
        "class_weight":      best_params["class_weight"],
        "bootstrap":         True,
        "random_state":      RANDOM_STATE,
        "n_jobs":            -1,
        "oob_score":         False,
    }

    cv_results, modelo_final, _ = _ejecuta_cv_RF(final_params, X_train, y_train, cv_folds)

    print(f"\n{'=============================================================================='}")
    print(f"  RESUMEN CROSS-VALIDATION (Optimizado) — FASE {fase}")
    print(f"{'=============================================================================='}")
    _imprime_resumen_cv(cv_results, cv_folds, fase)

    # Registro en MLflow
    with mlflow.start_run(run_name=f"Optuna_RandomForest_CV5_{fase}"):
        mlflow.set_tag("modelo", "Optimizado_Optuna")
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
        "fase":             fase,
        "model":            modelo_final,
        "best_params":      best_params,
        "best_f1_score_cv": best_f1_cv,
        "cv_results":       cv_results,
        "study":            study,
    }


# ==============================================================================
# FUNCIÓN PRINCIPAL DEL PIPELINE
# ==============================================================================

def modelado_RF(
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
    data_path    = Path(input_path)         if input_path         else DATA_PROCESSED_PATH
    fig_dir      = Path(output_dir_figures) if output_dir_figures else OUTPUT_DIR_FIGURES
    models_dir   = Path(output_dir_models)  if output_dir_models  else OUTPUT_DIR_MODELS

    _mlruns_path = Path(mlruns_dir).resolve() if mlruns_dir else MLRUNS_DIR.resolve()
    mlruns_uri   = _mlruns_path.as_uri()   # → file:///C:/... en Windows, file:///home/... en Linux

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
        print("==============================================================================")
        print("  CONFIGURACIÓN MLFLOW")
        print("==============================================================================")
        print(f"  Tracking URI  : {mlruns_uri}")
        print(f"  Experiment ID : {experiment.experiment_id if experiment else 'Nuevo'}")
        print(f"\n  Para visualizar resultados:\n    mlflow ui --port 5000")

    # ------------------------------------------------------------------
    # 1. Carga de datos
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "==============================================================================")
        print("  1. CARGA DE DATOS PREPROCESADOS")
        print("==============================================================================")

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
        print("\n" + "==============================================================================")
        print("  2. VARIABLES POR FASE TEMPORAL")
        print("==============================================================================")
        print(f"  T0 (Matrícula)   : {len(VARS_T0)} variables")
        print(f"  T1 (Fin 1er Sem) : {len(VARS_T1)} variables (+{len(VARS_T1) - len(VARS_T0)})")
        print(f"  T2 (Fin 2do Sem) : {len(VARS_T2)} variables (+{len(VARS_T2) - len(VARS_T1)})")

    # ------------------------------------------------------------------
    # 3. Split estratificado
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "==============================================================================")
        print("  3. SPLIT TRAIN / TEST  (80 / 20 estratificado)")
        print("==============================================================================")

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
    # 4-7. Loop por fase: preprocesamiento 
    # ------------------------------------------------------------------
    csv_path_rf = models_dir / "cv_summary_RF.csv"
    df_acumulado = pd.DataFrame()

    for fase in ["T0", "T1", "T2"]:
        if verbose:
            print("\n" + "==============================================================================")
            print(f"  FASE {fase}")
            print("==============================================================================")

        # Preprocesamiento
        X_tr, X_te, features, prep = preprocesamiento_RF(X_train, X_test, y_train, fase)

        if verbose:
            print(f"\n  Dimensiones post-preprocesamiento:")
            print(f"    Train : {X_tr.shape}  |  Test: {X_te.shape}  |  Features: {len(features)}")

        # --- Sin optimizacion ---
        results_base = entrena_RF(X_tr, y_train, fase, cv_folds, fig_dir)

        df_base = resumen_cv(results_base["cv_results"], fase, "RandomForest")
        df_acumulado = pd.concat([df_acumulado, df_base], ignore_index=True)
        df_acumulado.to_csv(csv_path_rf, index=False)

        # --- Optuna ---
        results_opt = entrena_RF_con_optuna(X_tr, y_train, fase, n_trials, cv_folds, fig_dir)

        if verbose:
            print(f"\n  Comparación F1-score — {fase}:")
            print(f"    Sin optimizacion : {results_base['cv_results']['test_f1'].mean():.4f}")
            print(f"    Optuna   : {results_opt['best_f1_score_cv']:.4f}")

        df_opt = resumen_cv(results_opt["cv_results"], fase, "RandomForest_opt")
        df_acumulado = pd.concat([df_acumulado, df_opt], ignore_index=True)
        df_acumulado.to_csv(csv_path_rf, index=False)

    # ------------------------------------------------------------------
    # 8. Resumen final RF
    # ------------------------------------------------------------------
    print("\n" + "==============================================================================")
    print("  RESUMEN FINAL — RANDOM FOREST (CROSS-VALIDATION)")
    print("==============================================================================")
    df_rf_final = pd.read_csv(csv_path_rf)
    print(df_rf_final.to_string(index=False))
    print(f"\n  Resultados guardados en: {csv_path_rf}")

    # ------------------------------------------------------------------
    # 9. Resumen comparativo global (RL + RF)
    # ------------------------------------------------------------------
    csv_path_rl = OUTPUT_DIR_MODELS_GLOBAL / "baseline_RL" / "cv_summary_RL.csv"
    csv_path_global = OUTPUT_DIR_MODELS_GLOBAL / "cv_summary_entrenamiento.csv"

    if csv_path_rl.exists():
        df_rl = pd.read_csv(csv_path_rl)
        df_global = pd.concat([df_rl, df_rf_final], ignore_index=True)
        df_global.to_csv(csv_path_global, index=False)
        print(f"\n  Resumen comparativo RL + RF guardado en: {csv_path_global}")
    else:
        if verbose:
            print(f"\n  Aviso: no se encontró {csv_path_rl}. "
                  f"El resumen comparativo global se generará cuando se ejecute la stage de RL.")



# Funcion principal
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de entrenamiento sin optimizacion - Random Forest"
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
        help="Directorio de salida para figuras (default: outputs/figures/modelado/RF)",
    )
    parser.add_argument(
        "--models", "-m",
        type=str,
        default=None,
        help="Directorio de salida para reportes CSV (default: outputs/models/RF)",
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

    modelado_RF(
        input_path=args.input,
        output_dir_figures=args.figures,
        output_dir_models=args.models,
        mlruns_dir=args.mlruns,
        n_trials=args.n_trials,
        cv_folds=args.cv_folds,
        verbose=not args.quiet,
    )

    print("\n" + "==============================================================================")
    print("  MODELADO RF COMPLETADO")
    print("==============================================================================")


if __name__ == "__main__":
    main()
