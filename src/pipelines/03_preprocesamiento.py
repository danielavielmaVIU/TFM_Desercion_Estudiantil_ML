#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
PREPROCESAMIENTO DE DATOS
================================================================================
Dataset: Predict Students' Dropout and Academic Success
Fuente: UCI Repository
Autor: maria Vielma
Fecha: Febrero 2026

Descripción:
    Script para preprocesar el dataset de deserción estudiantil.
    Incluye:
    - Creación de variables derivadas (feature engineering)
    - Agrupación de categorías por riesgo
    - Creación de target binario (Dropout vs No Dropout)
    - Eliminación de variables redundantes
    
    NO incluye escalado (se hace en modelado según algoritmo).

Uso con DVC:
    python src/data/3_preprocesamiento.py

Outputs:
    - data/processed/preprocessed_data.csv
    - outputs/figures/preprocesamiento/01_distribucion_target_binario.png
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from pathlib import Path

# ==============================================================================
# CONFIGURACIÓN DE RUTAS
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Ajustar según ubicación
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.clean_columns import clean_dataframe_columns
from src.utils.constants import (
    VARS_BINARIAS,
    VARS_CATEGORICAS_NOMINALES,
    VARS_CATEGORICAS_ORDINALES,
    VARS_NUMERICAS,
    TARGET,
    TARGET_VALUES,
    LABELS
)

# ==============================================================================
# CONFIGURACIÓN DE PATHS
# ==============================================================================
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "data.csv"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessed_data.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "preprocesamiento"

# Estilo de matplotlib
plt.style.use("seaborn-v0_8-whitegrid")


# ==============================================================================
# FUNCIONES DE TRANSFORMACIÓN
# ==============================================================================

def crear_is_single(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variable binaria indicando si el estudiante es soltero.
    
    Justificación: Estado civil puede influir en disponibilidad de tiempo
    y recursos para el estudio.
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con la columna 'marital_status'.
    
    Retorna
    -------
    pd.DataFrame
        DataFrame con nueva columna 'is_single'.
    """
    df['is_single'] = (df['marital_status'] == 1).astype(int)
    return df


def agrupar_application_mode(x: int) -> str:
    """
    Agrupa modalidad de aplicación en 3 categorías por nivel de riesgo.
    
    Basado en análisis de tasas de deserción por modalidad.
    
    Parámetros
    ----------
    x : int
        Código de modalidad de aplicación.
    
    Retorna
    -------
    str
        Nivel de riesgo: 'Alto_Riesgo', 'Riesgo_Medio', 'Bajo_Riesgo'.
    """
    # ALTO RIESGO (>40% deserción)
    if x in [39, 7, 42, 2, 26, 27]:
        return 'Alto_Riesgo'
    # RIESGO MEDIO (30-40% deserción)
    elif x in [43, 18, 51, 10]:
        return 'Riesgo_Medio'
    # BAJO RIESGO (<30% deserción)
    else:  # 1, 17, 44, 15, 16, 5, 53, 57
        return 'Bajo_Riesgo'


def crear_application_mode_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variable de riesgo basada en modalidad de aplicación.
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columna 'application_mode'.
    
    Retorna
    -------
    pd.DataFrame
        DataFrame con nueva columna 'application_mode_risk'.
    """
    df['application_mode_risk'] = df['application_mode'].apply(agrupar_application_mode)
    return df


def crear_is_over_23_entry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variable binaria para estudiantes que ingresaron como "Mayor de 23 años".
    
    Justificación: Los estudiantes que ingresan por la vía 'Mayor de 23 años' 
    (código 39) presentan una tasa de deserción del 55.4%, casi triple que los 
    estudiantes de 1ra fase general (20.2%). Este grupo representa el 17.7% del 
    dataset (785 estudiantes).
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columna 'application_mode'.
    
    Retorna
    -------
    pd.DataFrame
        DataFrame con nueva columna 'is_over_23_entry'.
    """
    df['is_over_23_entry'] = (df['application_mode'] == 39).astype(int)
    return df


def agrupar_previous_qualification_riesgo(x: int) -> str:
    """
    Agrupa cualificación previa en categorías por nivel de riesgo.
    
    Parámetros
    ----------
    x : int
        Código de cualificación previa.
    
    Retorna
    -------
    str
        Nivel de riesgo.
    """
    # ALTO RIESGO: Incompleta + Ed. superior previa
    if x in [9, 10, 14, 15, 2, 3, 19, 12, 5, 4, 6]:
        return 'Alto_Riesgo'
    # RIESGO MEDIO
    elif x in [38, 40]:
        return 'Riesgo_Medio'
    # BAJO RIESGO: Secundaria, técnicos
    else:
        return 'Bajo_Riesgo'


def crear_previous_qualification_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variable de riesgo basada en cualificación previa.
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columna 'previous_qualification'.
    
    Retorna
    -------
    pd.DataFrame
        DataFrame con nueva columna 'previous_qualification_risk'.
    """
    df['previous_qualification_risk'] = df['previous_qualification'].apply(
        agrupar_previous_qualification_riesgo
    )
    return df


def agrupar_parent_qualification(x: int) -> str:
    """
    Agrupa nivel educativo de padres en categorías jerárquicas.
    
    Parámetros
    ----------
    x : int
        Código de cualificación del padre/madre.
    
    Retorna
    -------
    str
        Nivel educativo agrupado.
    """
    if x == 34:
        return 'Desconocido'
    elif x in [35, 36, 20, 13, 25, 33, 31]:
        return 'Sin_Educacion'
    elif x == 37:
        return 'Basica_Baja'
    elif x in [38, 19, 11, 30, 26, 29]:
        return 'Basica_Media'
    elif x in [1, 12, 9, 10, 14, 15, 18, 22, 27]:
        return 'Secundaria'
    elif x in [2, 3, 4, 5, 6, 39, 40, 41, 42, 43, 44]:
        return 'Superior'
    else:
        return 'Desconocido'


def crear_parent_qualification_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables de nivel educativo para ambos padres.
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columnas 'mothers_qualification' y 'fathers_qualification'.
    
    Retorna
    -------
    pd.DataFrame
        DataFrame con nuevas columnas de nivel educativo.
    """
    df['mothers_qualification_level'] = df['mothers_qualification'].apply(
        agrupar_parent_qualification
    )
    df['fathers_qualification_level'] = df['fathers_qualification'].apply(
        agrupar_parent_qualification
    )
    return df


def agrupar_parent_occupation(x: int) -> str:
    """
    Agrupa ocupación de padres en categorías.
    
    Parámetros
    ----------
    x : int
        Código de ocupación del padre/madre.
    
    Retorna
    -------
    str
        Ocupación agrupada.
    """
    if x in [90, 99]:
        return 'Sin_Info'
    elif x == 0:
        return 'Estudiante'
    elif x in [1, 2, 3]:
        return 'Profesional'
    else:
        return 'Otro_Trabajo'


def crear_parent_occupation_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables de nivel ocupacional para ambos padres.
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columnas 'mothers_occupation' y 'fathers_occupation'.
    
    Retorna
    -------
    pd.DataFrame
        DataFrame con nuevas columnas de nivel ocupacional.
    """
    df['mothers_occupation_level'] = df['mothers_occupation'].apply(
        agrupar_parent_occupation
    )
    df['fathers_occupation_level'] = df['fathers_occupation'].apply(
        agrupar_parent_occupation
    )
    return df


def crear_has_unknown_parent_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variable indicando si hay información desconocida de los padres.
    
    Justificación: 
    - Cuando mothers/fathers_qualification = 34 ('Desconocido'), la tasa 
      de deserción está sobre el 70%.
    - Cuando mothers/fathers_occupation = 90, 99 o 0, la tasa de deserción 
      está entre el 64% y 77%.
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columnas de cualificación y ocupación de padres.
    
    Retorna
    -------
    pd.DataFrame
        DataFrame con nueva columna 'has_unknown_parent_info'.
    """
    df['has_unknown_parent_info'] = (
        (df['mothers_qualification'] == 34) |
        (df['fathers_qualification'] == 34) |
        (df['mothers_occupation'].isin([90, 99, 0])) |
        (df['fathers_occupation'].isin([90, 99, 0]))
    ).astype(int)
    return df


def crear_target_binario(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variable target binaria para clasificación.
    
    1 = Dropout (desertor)
    0 = No Dropout (Graduate + Enrolled)
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columna 'target'.
    
    Retorna
    -------
    pd.DataFrame
        DataFrame con nueva columna 'target_binario'.
    """
    df['target_binario'] = (df['target'] == 'Dropout').astype(int)
    return df


def eliminar_variables_redundantes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina variables que ya fueron agrupadas o son redundantes.
    
    Variables eliminadas:
    - marital_status: reemplazada por is_single
    - application_mode: reemplazada por application_mode_risk
    - previous_qualification: reemplazada por previous_qualification_risk
    - mothers/fathers_qualification: reemplazadas por *_level
    - mothers/fathers_occupation: reemplazadas por *_level
    - nacionality: 99% de estudiantes son de Portugal
    - target: reemplazada por target_binario
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con todas las variables.
    
    Retorna
    -------
    pd.DataFrame
        DataFrame sin variables redundantes.
    """
    variables_a_eliminar = [
        "marital_status",
        "application_mode",
        "previous_qualification",
        "mothers_qualification",
        "fathers_qualification",
        "mothers_occupation",
        "fathers_occupation",
        "nacionality",
        "target"
    ]
    
    # Eliminar solo si existen en el dataframe
    cols_to_drop = [c for c in variables_a_eliminar if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    return df


def graficar_distribucion_target(
    df: pd.DataFrame,
    output_dir: Path,
    verbose: bool = True
) -> None:
    """
    Genera gráfico de distribución del target binario.
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columna 'target_binario'.
    output_dir : Path
        Directorio para guardar la figura.
    verbose : bool
        Si True, muestra información de guardado.
    """
    prop = df['target_binario'].value_counts(normalize=True).sort_index()
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    prop.plot(
        kind='bar',
        color=['#2ca02c', '#E74C3C'],
        ax=ax
    )
    
    # Agregar etiquetas de porcentaje sobre las barras
    for i, v in enumerate(prop.values):
        ax.text(
            i,
            v + 0.01,
            f"{v:.1%}",
            ha='center',
            fontsize=8
        )
    
    ax.set_title("Distribución de la variable objetivo (clase binaria)", fontsize=12)
    ax.set_xlabel("Clase (0=No Dropout, 1=Dropout)", fontsize=10)
    ax.set_ylabel("Proporción", fontsize=10)
    ax.set_xticklabels(['No Dropout', 'Dropout'], rotation=0)
    
    plt.tight_layout()
    
    # Guardar figura
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / "01_distribucion_target_binario.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"Figura guardada en: {filepath}")


def imprimir_resumen(df: pd.DataFrame) -> None:
    """
    Imprime resumen del dataset preprocesado.
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame preprocesado.
    """
    print("\n" + "=" * 70)
    print("RESUMEN DEL PREPROCESAMIENTO")
    print("=" * 70)
    
    # Dimensiones
    print(f"\nDIMENSIONES:")
    print(f"   • Filas: {df.shape[0]:,}")
    print(f"   • Columnas: {df.shape[1]}")
    
    # Columnas finales
    print(f"\nCOLUMNAS FINALES ({df.shape[1]}):")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
    
    # Tipos de datos
    print("\nTIPOS DE DATOS:")
    print(df.dtypes.value_counts().to_string())
    
    # Missing values
    missing_total = df.isnull().sum().sum()
    print(f"\nVALORES FALTANTES: {missing_total}")
    if missing_total > 0:
        print("   Columnas con valores faltantes:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
    else:
        print("No hay valores faltantes")
    
    # Target binario
    print("\nTARGET BINARIO:")
    print(df['target_binario'].value_counts().to_string())
    ratio = df['target_binario'].value_counts()[0] / df['target_binario'].value_counts()[1]
    print(f"\n   Ratio (No Desertor / Desertor): {ratio:.2f}:1")
    print(f"   Desbalance: {'Moderado' if ratio < 3 else 'Alto'}")


# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def preprocesar_datos(
    input_path: str = None,
    output_path: str = None,
    figures_dir: str = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de preprocesamiento.
    
    Parámetros
    ----------
    input_path : str, opcional
        Ruta al archivo CSV de entrada.
    output_path : str, opcional
        Ruta para guardar el CSV procesado.
    figures_dir : str, opcional
        Directorio para guardar figuras.
    verbose : bool
        Si True, imprime información durante la ejecución.
    
    Retorna
    -------
    pd.DataFrame
        DataFrame preprocesado.
    """
    input_path = Path(input_path) if input_path else DATA_RAW_PATH
    output_path = Path(output_path) if output_path else DATA_PROCESSED_PATH
    figures_dir = Path(figures_dir) if figures_dir else FIGURES_DIR
    
    # ==========================================================================
    # 1. CARGAR DATASET
    # ==========================================================================
    if verbose:
        print("=" * 70)
        print("PREPROCESAMIENTO DE DATOS")
        print("=" * 70)
        print(f"\nCargando dataset desde: {input_path}")
    
    df_raw = pd.read_csv(input_path, delimiter=';')
    df = df_raw.copy()
    df = clean_dataframe_columns(df)
    
    if verbose:
        print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # ==========================================================================
    # 2. FEATURE ENGINEERING
    # ==========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("Aplicando transformaciones...")
        print("-" * 70)
    
    # 2.1 Estado civil → is_single
    df = crear_is_single(df)
    if verbose:
        print("Creada: is_single")
    
    # 2.2 Modalidad de aplicación → application_mode_risk
    df = crear_application_mode_risk(df)
    if verbose:
        print("Creada: application_mode_risk")
    
    # 2.3 Mayor de 23 años → is_over_23_entry
    df = crear_is_over_23_entry(df)
    if verbose:
        print("Creada: is_over_23_entry")
    
    # 2.4 Cualificación previa → previous_qualification_risk
    df = crear_previous_qualification_risk(df)
    if verbose:
        print("Creada: previous_qualification_risk")
    
    # 2.5 Educación de padres → *_qualification_level
    df = crear_parent_qualification_levels(df)
    if verbose:
        print("Creadas: mothers_qualification_level, fathers_qualification_level")
    
    # 2.6 Ocupación de padres → *_occupation_level
    df = crear_parent_occupation_levels(df)
    if verbose:
        print("Creadas: mothers_occupation_level, fathers_occupation_level")
    
    # 2.7 Información desconocida de padres → has_unknown_parent_info
    df = crear_has_unknown_parent_info(df)
    if verbose:
        print("Creada: has_unknown_parent_info")
    
    # 2.8 Target binario
    df = crear_target_binario(df)
    if verbose:
        print("Creada: target_binario")
    
    # ==========================================================================
    # 3. ELIMINAR VARIABLES REDUNDANTES
    # ==========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("Eliminando variables redundantes...")
        print("-" * 70)
    
    n_cols_antes = df.shape[1]
    df = eliminar_variables_redundantes(df)
    n_cols_despues = df.shape[1]
    
    if verbose:
        print(f"Variables eliminadas: {n_cols_antes - n_cols_despues}")
    
    # ==========================================================================
    # 4. GENERAR VISUALIZACIONES
    # ==========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("Generando visualizaciones...")
        print("-" * 70)
    
    graficar_distribucion_target(df, figures_dir, verbose)
    
    # ==========================================================================
    # 5. GUARDAR DATASET PROCESADO
    # ==========================================================================
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    if verbose:
        print(f"\nDataset procesado guardado en: {output_path}")
        imprimir_resumen(df)
    
    return df


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    """Entry point para ejecución desde línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Preprocesamiento de datos para el dataset de deserción estudiantil"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Ruta al archivo CSV de entrada"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Ruta para guardar el CSV procesado"
    )
    parser.add_argument(
        "--figures", "-f",
        type=str,
        default=None,
        help="Directorio para guardar figuras"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Ejecutar sin mensajes de progreso"
    )
    
    args = parser.parse_args()
    
    preprocesar_datos(
        input_path=args.input,
        output_path=args.output,
        figures_dir=args.figures,
        verbose=not args.quiet
    )
    
    print("\n" + "=" * 70)
    print("PREPROCESAMIENTO COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()
