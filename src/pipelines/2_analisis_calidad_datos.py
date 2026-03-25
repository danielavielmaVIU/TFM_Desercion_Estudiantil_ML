#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ANÁLISIS DE CALIDAD DE DATOS

import pandas as pd
import numpy as np
import os
import sys
import argparse
from pathlib import Path

# Configuración de rutas

PROJECT_ROOT = Path(__file__).resolve().parents[2]  
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

# CONFIGURACIÓN DE PATHS
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables" / "calidad_datos"



# FUNCIONES DE SCORING
def score_distribucion(df: pd.DataFrame, col: str) -> float:
    skew = abs(df[col].skew())
    kurt = abs(df[col].kurtosis())

    score = 100.0
    score -= min(skew * 10, 40)  # penalización por skew (máx 40 puntos)
    score -= min(kurt * 5, 40)   # penalización por curtosis (máx 40 puntos)

    return max(score, 0)


def score_categorias_raras(df: pd.DataFrame, col: str, umbral: int = 10) -> int:
    conteos = df[col].value_counts()
    n_raras = (conteos < umbral).sum()

    if n_raras == 0:
        return 100
    elif n_raras <= 2:
        return 80
    elif n_raras <= 5:
        return 60
    else:
        return 30


def calcular_score_dominio(df: pd.DataFrame) -> dict:
    errores_dominio = {}

    for col, mapping in LABELS.items():
        valores_validos = set(mapping.keys())
        valores_actuales = set(df[col].unique())
        fuera = valores_actuales - valores_validos

        if len(fuera) > 0:
            errores_dominio[col] = list(fuera)

    score_dominio = {}
    for col in df.columns:
        if col not in errores_dominio:
            score_dominio[col] = 100
        else:
            n = len(errores_dominio[col])
            score_dominio[col] = max(0, 100 - 20 * n)

    return score_dominio, errores_dominio


def calcular_outliers_iqr(df: pd.DataFrame) -> tuple:
    outlier_report = []
    numericas = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numericas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        pct = round(n_outliers / len(df) * 100, 2)

        outlier_report.append([col, n_outliers, pct])

    tabla_outliers = pd.DataFrame(
        outlier_report, 
        columns=["Variable", "Outliers_IQR", "Pct_IQR"]
    )

    # Calcular score de rangos
    score_rangos = {}
    for _, row in tabla_outliers.iterrows():
        score_rangos[row["Variable"]] = max(0, 100 - row["Pct_IQR"] * 1.2)

    # Las no-numéricas reciben 100
    for col in df.columns:
        score_rangos.setdefault(col, 100)

    return tabla_outliers, score_rangos


def clasificar_calidad(score: float) -> str:
    if score >= 90:
        return "Excelente"
    elif score >= 80:
        return "Muy Buena"
    elif score >= 70:
        return "Aceptable"
    elif score >= 60:
        return "Baja"
    else:
        return "Crítica"


# Función principal
def analizar_calidad_datos(
    input_path: str = None,
    output_dir: str = None,
    verbose: bool = True
) -> pd.DataFrame:

    input_path = Path(input_path) if input_path else DATA_RAW_PATH
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    
    # Crear directorio de salida si no existe
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. CARGAR DATASET
    if verbose:
        print("--------------------------------------------------------------------------------")
        print("ANÁLISIS DE CALIDAD DE DATOS")
        print("--------------------------------------------------------------------------------")
        print(f"\nCargando dataset desde: {input_path}")
    
    df = pd.read_csv(input_path, delimiter=';')
    df = clean_dataframe_columns(df)
    
    if verbose:
        print(f"Dataset cargado correctamente")
        print(f"  - Filas: {df.shape[0]}")
        print(f"  - Columnas: {df.shape[1]}")
        print(f"  - Celdas totales: {df.shape[0] * df.shape[1]:,}")
    
    # 2. COMPLETITUD - Valores Nulos
    if verbose:
        print("\n" + "--------------------------------------------------------------------------------")
        print("2. COMPLETITUD - Valores Nulos")
        print("--------------------------------------------------------------------------------")
    
    nulls = df.isnull().sum()
    nulls_pct = (nulls / len(df)) * 100
    
    if verbose:
        print(f"Total de valores nulos en dataset: {nulls.sum()}")
    
    # 3. CONSISTENCIA - Duplicados
    if verbose:
        print("\n" + "--------------------------------------------------------------------------------")
        print("3. CONSISTENCIA - Duplicados")
        print("--------------------------------------------------------------------------------")
    
    duplicates = df.duplicated().sum()
    pct_dup = duplicates / len(df) * 100
    
    if verbose:
        print(f"Registros duplicados: {duplicates}")
        print(f"Porcentaje de filas duplicadas: {pct_dup:.2f}%")
    
    # 4. EXACTITUD - Validación de dominio
    if verbose:
        print("\n" + "--------------------------------------------------------------------------------")
        print("4. EXACTITUD - Validación de categorías fuera de dominio")
        print("--------------------------------------------------------------------------------")
    
    score_dominio, errores_dominio = calcular_score_dominio(df)
    
    if verbose:
        if len(errores_dominio) == 0:
            print("No existen valores fuera del dominio definido.")
        else:
            print("Valores fuera de dominio encontrados:")
            for variable, valores in errores_dominio.items():
                print(f"  - {variable}: {valores}")
    
    # 5. OUTLIERS - Método IQR
    if verbose:
        print("\n" + "--------------------------------------------------------------------------------")
        print("5. OUTLIERS - Método IQR")
        print("--------------------------------------------------------------------------------")
    
    tabla_outliers, score_rangos = calcular_outliers_iqr(df)
    
    if verbose:
        # Mostrar solo variables con outliers > 0
        outliers_con_datos = tabla_outliers[tabla_outliers["Outliers_IQR"] > 0]
        print(f"Variables con outliers: {len(outliers_con_datos)}")
        if len(outliers_con_datos) > 0:
            print(outliers_con_datos.to_string(index=False))
    
    # 6. ÍNDICE GLOBAL DE CALIDAD
    if verbose:
        print("\n" + "--------------------------------------------------------------------------------")
        print("6. ÍNDICE GLOBAL DE CALIDAD DEL DATASET")
        print("\n" + "--------------------------------------------------------------------------------")
    
    # Crear base del índice con todas las columnas del dataset
    metricas = pd.DataFrame(index=df.columns)
    
    # COMPLETITUD
    metricas["Valores_nulos"] = 100 - nulls_pct
    
    # CONSISTENCIA
    metricas["Duplicados"] = 100 - pct_dup
    
    # Score de distribuciones sesgadas (solo numéricas)
    metricas["Score_sesgo"] = [
        score_distribucion(df, col) if col in VARS_NUMERICAS else np.nan
        for col in df.columns
    ]
    
    # Score de categorías raras (solo categóricas nominales)
    metricas["Score_categorias_raras"] = [
        score_categorias_raras(df, col) if col in VARS_CATEGORICAS_NOMINALES else np.nan
        for col in df.columns
    ]
    
    # EXACTITUD
    metricas["Exactitud_dominio"] = metricas.index.map(score_dominio)
    metricas["Exactitud_rangos"] = metricas.index.map(score_rangos).round(2)
    
    # Score global ponderado (pesos iguales)
    metricas["Score_Calidad"] = (
        0.1666 * metricas["Valores_nulos"] +
        0.1666 * metricas["Duplicados"] +
        0.1666 * metricas["Score_sesgo"].fillna(100) +
        0.1666 * metricas["Score_categorias_raras"].fillna(100) +
        0.1666 * metricas["Exactitud_dominio"] +
        0.1666 * metricas["Exactitud_rangos"]
    ).round(2)
    
    # Redondear columnas
    metricas = metricas.round({
        "Valores_nulos": 2,
        "Duplicados": 2,
        "Score_sesgo": 2,
        "Score_categorias_raras": 2,
        "Exactitud_dominio": 2,
        "Exactitud_rangos": 2,
    })
    
    # Clasificación de calidad
    metricas["Nivel_Calidad"] = metricas["Score_Calidad"].apply(clasificar_calidad)
    
    # Ordenar por score
    metricas_sorted = metricas.sort_values("Score_Calidad")
    
    # 7. GUARDAR RESULTADOS
    
    output_file = output_dir / "indice_calidad_dataset.csv"
    metricas_sorted.to_csv(output_file, index=True)
    
    if verbose:
        print(f"\nResultados guardados en: {output_file}")
        print(f"\nResumen de calidad:")
        print(f"  - Score promedio: {metricas['Score_Calidad'].mean():.2f}")
        print(f"  - Score mínimo: {metricas['Score_Calidad'].min():.2f}")
        print(f"  - Score máximo: {metricas['Score_Calidad'].max():.2f}")
        print(f"\nDistribución por nivel de calidad:")
        print(metricas["Nivel_Calidad"].value_counts().to_string())
    
    return metricas_sorted


# Funcion principal
def main():
    parser = argparse.ArgumentParser(
        description="Análisis de calidad de datos para el dataset de deserción estudiantil"
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
        help="Directorio para guardar outputs"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Ejecutar sin mensajes de progreso"
    )
    
    args = parser.parse_args()
    
    analizar_calidad_datos(
        input_path=args.input,
        output_dir=args.output,
        verbose=not args.quiet
    )
    
    print("\n" + "================================================================================")
    print("\n" + "--------------------------------------------------------------------------------")
    print("Análisis completado")
    print("================================================================================")


if __name__ == "__main__":
    main()
