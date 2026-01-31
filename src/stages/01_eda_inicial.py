"""
==============================================================================
01_eda_inicial.py - Análisis Exploratorio de Datos
==============================================================================
Dataset: Predict Students' Dropout and Academic Success
Fuente: UCI Repository
Fecha: Enero 2026

Este script realiza el EDA completo del dataset y genera:
- Figuras en outputs/figures/EDA/
- Tablas resumen en outputs/tables/

Ejecutar: python src/stages/01_eda_inicial.py
==============================================================================
"""

import sys
import os
import warnings
import math

# Configurar path del proyecto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================
warnings.filterwarnings("ignore", message="Glyph.*missing")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (20, 20)
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

# Rutas de salida
BASE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs/figures/EDA")
TABLES_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs/tables")

# Paleta de colores para target
PALETTE_TARGET = {
    "Dropout": "#E74C3C",   
    "Enrolled": "#1F77B4",  
    "Graduate": "#2CA02C"   
}


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================
def create_output_dirs():
    """Crea los directorios de salida necesarios."""
    dirs = [
        os.path.join(BASE_OUTPUT_DIR, "1_numericas"),
        os.path.join(BASE_OUTPUT_DIR, "2_binarias"),
        os.path.join(BASE_OUTPUT_DIR, "3_categoricas"),
        os.path.join(BASE_OUTPUT_DIR, "4_target"),
        TABLES_OUTPUT_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def load_data():
    """Carga y limpia el dataset."""
    data_path = os.path.join(PROJECT_ROOT, "data/raw/data.csv")
    df = pd.read_csv(data_path, delimiter=';')
    df = clean_dataframe_columns(df)
    return df


# =============================================================================
# 1. CARGA Y DIMENSIÓN DEL DATASET
# =============================================================================
def section_1_carga_dimension(df):
    """Sección 1: Carga y dimensión del dataset."""
    print("=" * 80)
    print("1. CARGA Y DIMENSIÓN DEL DATASET")
    print("=" * 80)
    print(f"\n Dataset cargado")
    print(f" - Observaciones (filas): {df.shape[0]:,}")
    print(f" - Variables (columnas): {df.shape[1]}, incluye columna Target")
    print(f" - Total de celdas: {df.shape[0] * df.shape[1]:,}")
    print("\n" + "-" * 80)
    print("Primeras 5 filas del dataset:")
    print("-" * 80)
    print(df.head())


# =============================================================================
# 2. LISTADO DE VARIABLES, NULOS Y DUPLICADOS
# =============================================================================
def section_2_listado_variables(df):
    """Sección 2: Listado de variables, verificación de nulos y duplicados."""
    print("\n" + "=" * 80)
    print("2. LISTADO DE VARIABLES DEL DATASET, VERIFICACION DE NULOS Y DUPLICADOS")
    print("=" * 80)

    # Lista las variables
    print(f"\n{'#':<4} {'Variable':<55} {'Tipo':<10}")
    print("-" * 80)
    for i, (col, dtype) in enumerate(zip(df.columns, df.dtypes), 1):
        print(f"{i:<4} {col:<55} {str(dtype):<10}")

    # Verificar valores nulos
    print("\n" + "-" * 80)
    print(" Verificación de Valores Nulos:")
    print("-" * 80)
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()

    if total_nulls == 0:
        print(" No hay valores nulos en el dataset")
    else:
        print(f" Total de valores nulos: {total_nulls}")
        print(null_counts[null_counts > 0])

    # Verificar Duplicados
    print("\n" + "-" * 80)
    print(" Verificación de Valores duplicados:")
    print("-" * 80)
    duplicates = df.duplicated().sum()
    if duplicates == 0:
        print(" No hay registros duplicados")
    else:
        print(f" Registros duplicados: {duplicates}")


# =============================================================================
# 3. CLASIFICACIÓN DE VARIABLES POR TIPO
# =============================================================================
def section_3_clasificacion_variables():
    """Sección 3: Clasificación de variables por tipo."""
    print("\n" + "=" * 80)
    print("3.0. CLASIFICACIÓN DE VARIABLES POR TIPO")
    print("=" * 80)

    total_classified = (len(VARS_BINARIAS) + len(VARS_CATEGORICAS_NOMINALES) + 
                        len(VARS_CATEGORICAS_ORDINALES) + len(VARS_NUMERICAS) + 
                        len(TARGET))

    print(f"\n CANTIDAD DE VARIABLES NUMÉRICAS: {len(VARS_NUMERICAS)} variables")
    print(f"\n CANTIDAD DE VARIABLES CATEGÓRICAS BINARIAS: {len(VARS_BINARIAS)} variables")
    print(f"\n CANTIDAD DE VARIABLES CATEGÓRICAS NOMINALES: {len(VARS_CATEGORICAS_NOMINALES)} variables")
    print(f"\n CANTIDAD DE VARIABLES CATEGÓRICAS ORDINALES: {len(VARS_CATEGORICAS_ORDINALES)} variable")
    print(f"\n TARGET: {len(TARGET)} variable (Clases: {', '.join(TARGET_VALUES)})")
    print(f"\n TOTAL VARIABLES CLASIFICADAS: {total_classified}")


# =============================================================================
# 3.1 ANÁLISIS DE VARIABLES NUMÉRICAS
# =============================================================================
def section_3_1_variables_numericas(df):
    """Sección 3.1: Análisis de variables numéricas."""
    print("\n" + "=" * 80)
    print("3.1. ANÁLISIS DE VARIABLES NUMÉRICAS")
    print("=" * 80)
    
    output_dir = os.path.join(BASE_OUTPUT_DIR, "1_numericas")
    
    # Estadísticas descriptivas
    stats_df = df[VARS_NUMERICAS].describe().T
    print("\nEstadísticas descriptivas:")
    print(stats_df)
    
    # Guardar tabla
    stats_df.to_csv(os.path.join(TABLES_OUTPUT_DIR, "eda_numeric_stats.csv"))
    
    # --- Histogramas ---
    n_vars = len(VARS_NUMERICAS)
    n_cols = 5
    n_rows = (n_vars // n_cols) + (1 if n_vars % n_cols != 0 else 0)

    df[VARS_NUMERICAS].hist(bins=20, figsize=(20, 16), grid=False, layout=(n_rows, n_cols))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_distribucion_variables_numericas.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("\n- Histogramas guardados: 01_distribucion_variables_numericas.png")

    # --- Boxplots ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 14))
    axes = axes.flatten()

    for i, col in enumerate(VARS_NUMERICAS):
        ax = axes[i]
        sns.boxplot(y=df[col], ax=ax, color='steelblue', width=0.5)
        ax.set_title(col, fontsize=10, fontweight='bold')
        ax.set_ylabel('')
        
        # Información de outliers
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        pct = (n_outliers / len(df)) * 100
        
        ax.annotate(f'n={n_outliers} ({pct:.1f}%)', xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=8, color='black')

    for j in range(len(VARS_NUMERICAS), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_boxplot_variables_numericas.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("- Boxplots guardados: 02_boxplot_variables_numericas.png")

    # --- Matriz de correlación ---
    corr_matrix = df[VARS_NUMERICAS].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8, 'label': 'Correlación'},
                annot_kws={'size': 8}, ax=ax)

    ax.set_title('Matriz de Correlación - Variables Numéricas', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_correlation_matrix_variables_numericas.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("- Matriz de correlación guardada: 03_correlation_matrix_variables_numericas.png")
    
    # Guardar matriz de correlación
    corr_matrix.to_csv(os.path.join(TABLES_OUTPUT_DIR, "eda_correlation_matrix.csv"))

    # --- Boxplots por Target ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 16))
    axes = axes.flatten()

    for i, col in enumerate(VARS_NUMERICAS):
        ax = axes[i]
        sns.boxplot(x='target', y=col, data=df, ax=ax, 
                    hue='target', palette=PALETTE_TARGET, width=0.6, legend=False)
        ax.set_title(col, fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='x', labelsize=8)

    for j in range(len(VARS_NUMERICAS), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Distribución de Variables Numéricas por Target', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_boxplot_variables_numericas_by_target.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("- Boxplots por target guardados: 04_boxplot_variables_numericas_by_target.png")


# =============================================================================
# 3.2 ANÁLISIS DE VARIABLES BINARIAS
# =============================================================================
def section_3_2_variables_binarias(df):
    """Sección 3.2: Análisis de variables binarias."""
    print("\n" + "=" * 80)
    print("3.2. ANÁLISIS DE VARIABLES BINARIAS")
    print("=" * 80)
    
    output_dir = os.path.join(BASE_OUTPUT_DIR, "2_binarias")
    
    # Crear DataFrame resumen
    resume_vars = []
    for var in VARS_BINARIAS:
        counts = df[var].value_counts().sort_index()
        n_0 = counts.get(0, 0)
        n_1 = counts.get(1, 0)
        pct_0 = (n_0 / len(df)) * 100
        pct_1 = (n_1 / len(df)) * 100
        
        labels = LABELS.get(var, {0: 'No', 1: 'Sí'})
        
        resume_vars.append({
            'Variable': var,
            'Label_0': labels[0],
            'N_0': n_0,
            '%_0': pct_0,
            'Label_1': labels[1],
            'N_1': n_1,
            '%_1': pct_1
        })

    binary_df = pd.DataFrame(resume_vars)

    print("\nDistribución de Variables Binarias:")
    print("-" * 96)
    print(f"{'Variable':<30} {'Valor=0':<15} {'N':>7} {'%':>7}   {'Valor=1':<15} {'N':>7} {'%':>7}")
    print("-" * 96)

    for _, row in binary_df.iterrows():
        print(f"{row['Variable']:<30} {row['Label_0']:<15} {row['N_0']:>7,} {row['%_0']:>6.1f}%   {row['Label_1']:<15} {row['N_1']:>7,} {row['%_1']:>6.1f}%")

    # Guardar tabla
    binary_df.to_csv(os.path.join(TABLES_OUTPUT_DIR, "eda_binary_vars.csv"), index=False)

    # --- Gráfico de barras univariado ---
    n_vars = len(VARS_BINARIAS)
    n_cols = 3
    n_rows = math.ceil(n_vars / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(VARS_BINARIAS):
        ax = axes[i]
        counts = df[col].value_counts().sort_index()
        sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, palette="Blues_r")
        ax.set_title(f"{col}", fontsize=11, fontweight='bold')
        ax.set_ylabel("Frecuencia")
        ax.set_xlabel("")
        
    for j in range(n_vars, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_Grafico_barras_variables_binarias.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("\n- Gráfico de barras guardado: 05_Grafico_barras_variables_binarias.png")

    # --- Gráfico bivariado por target ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(VARS_BINARIAS):
        ax = axes[i]
        ctab = pd.crosstab(df[col], df['target'], normalize='index')
        ctab.plot(
            kind='bar',
            stacked=True,
            ax=ax,
            color=[PALETTE_TARGET[c] for c in ctab.columns]
        )
        ax.set_title(f"{col} vs target", fontsize=11, fontweight='bold')
        ax.set_ylabel("Proporción")
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')

    for j in range(n_vars, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_Grafico_variables_binarias_by_target.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("- Gráfico bivariado guardado: 06_Grafico_variables_binarias_by_target.png")


# =============================================================================
# 3.3 ANÁLISIS DE VARIABLES CATEGÓRICAS (NOMINALES Y ORDINALES)
# =============================================================================
def section_3_3_variables_categoricas(df):
    """Sección 3.3: Análisis de variables categóricas nominales y ordinales."""
    print("\n" + "=" * 80)
    print("3.3. ANÁLISIS DE VARIABLES CATEGÓRICAS (NOMINALES Y ORDINALES)")
    print("=" * 80)
    
    output_dir = os.path.join(BASE_OUTPUT_DIR, "3_categoricas")
    VARS_CATEGORICAS = VARS_CATEGORICAS_NOMINALES + VARS_CATEGORICAS_ORDINALES
    
    all_summaries = []
    
    for col in VARS_CATEGORICAS:
        print(f"\n ---- Variable: {col.upper()} ----")
        
        # --- Gráfico Univariado ---
        plt.figure(figsize=(10, 4))
        df[col].value_counts().plot(kind='bar')
        plt.title(f"Distribución de {col}")
        plt.xlabel(col)
        plt.ylabel("Frecuencia")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"07_univariado_{col}.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Tabla resumen univariado
        summary = pd.DataFrame({
            'Variable': col,
            'Valor': df[col].value_counts().index,
            'N': df[col].value_counts().values,
            'Porcentaje': (df[col].value_counts(normalize=True) * 100).round(2).values
        })
        
        if col in LABELS:
            summary['Descripcion'] = summary['Valor'].map(LABELS[col])
        
        all_summaries.append(summary)

        # --- Gráfico Bivariado ---
        ctab = pd.crosstab(df[col], df['target'], normalize="index")
        
        if len(ctab) > 8:
            figsize = (10, 6)
            kind = "barh"
        else:
            figsize = (8, 4)
            kind = "bar"

        fig, ax = plt.subplots(figsize=figsize)
        ctab.plot(
            kind=kind,
            stacked=True,
            color=list(PALETTE_TARGET.values()),
            ax=ax,
            edgecolor="black"
        )

        plt.title(f"{col} vs target (proporciones)", fontsize=12)

        if kind == "bar":
            ax.set_xlabel(col)
            ax.set_ylabel("Proporción")
            plt.xticks(rotation=25, ha="right", fontsize=8)
        else:
            ax.set_xlabel("Proporción")
            ax.set_ylabel(col)
            plt.yticks(fontsize=8)

        ax.legend(title="Target", bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"08_bivariado_{col}.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    # Guardar todas las tablas resumen
    all_summaries_df = pd.concat(all_summaries, ignore_index=True)
    all_summaries_df.to_csv(os.path.join(TABLES_OUTPUT_DIR, "eda_categorical_vars.csv"), index=False)
    
    print(f"\n- Gráficos categóricos guardados en: {output_dir}")
    print(f"- Tabla resumen guardada: eda_categorical_vars.csv")


# =============================================================================
# 3.4 ANÁLISIS DE VARIABLE TARGET
# =============================================================================
def section_3_4_variable_target(df):
    """Sección 3.4: Análisis de la variable objetivo (target)."""
    print("\n" + "=" * 80)
    print("3.4. ANÁLISIS DE VARIABLE TARGET")
    print("=" * 80)
    
    output_dir = os.path.join(BASE_OUTPUT_DIR, "4_target")
    
    # Distribución
    print("\nDistribución de la variable objetivo:")
    print(df['target'].value_counts())
    print("\nProporciones:")
    print(df['target'].value_counts(normalize=True).round(4) * 100)
    
    # Gráfico
    plt.figure(figsize=(5, 4))
    df['target'].value_counts(normalize=True).plot(
        kind='bar',
        color=['#E74C3C', '#2ca02c', '#1f77b4']
    )
    plt.title("Distribución de la variable objetivo", fontsize=12)
    plt.xlabel("Clase", fontsize=10)
    plt.ylabel("Proporción", fontsize=10)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "09_distribucion_target.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n- Gráfico de target guardado: 09_distribucion_target.png")
    
    # Guardar resumen
    target_summary = pd.DataFrame({
        'Clase': df['target'].value_counts().index,
        'N': df['target'].value_counts().values,
        'Porcentaje': (df['target'].value_counts(normalize=True) * 100).round(2).values
    })
    target_summary.to_csv(os.path.join(TABLES_OUTPUT_DIR, "eda_target_distribution.csv"), index=False)


# =============================================================================
# GENERAR RESUMEN GENERAL (para DVC)
# =============================================================================
def generate_eda_summary(df):
    """Genera archivo resumen del EDA para DVC."""
    summary = {
        'n_rows': df.shape[0],
        'n_cols': df.shape[1],
        'n_nulls': df.isnull().sum().sum(),
        'n_duplicates': df.duplicated().sum(),
        'n_numeric_vars': len(VARS_NUMERICAS),
        'n_binary_vars': len(VARS_BINARIAS),
        'n_nominal_vars': len(VARS_CATEGORICAS_NOMINALES),
        'n_ordinal_vars': len(VARS_CATEGORICAS_ORDINALES),
        'target_distribution': df['target'].value_counts().to_dict()
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(TABLES_OUTPUT_DIR, "eda_summary.csv"), index=False)
    print(f"\n- Resumen EDA guardado: eda_summary.csv")


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================
def main():
    """Ejecuta el pipeline completo de EDA."""
    print("\n" + "=" * 80)
    print(" ANÁLISIS EXPLORATORIO DE DATOS - TFM DESERCIÓN ESTUDIANTIL")
    print("=" * 80)
    
    # Crear directorios
    create_output_dirs()
    
    # Cargar datos
    print("\nCargando datos...")
    df = load_data()
    
    # Ejecutar todas las secciones
    section_1_carga_dimension(df)
    section_2_listado_variables(df)
    section_3_clasificacion_variables()
    section_3_1_variables_numericas(df)
    section_3_2_variables_binarias(df)
    section_3_3_variables_categoricas(df)
    section_3_4_variable_target(df)
    
    # Generar resumen para DVC
    generate_eda_summary(df)
    
    print("\n" + "=" * 80)
    print(" EDA COMPLETADO")
    print("=" * 80)
    print(f"\n Figuras guardadas en: {BASE_OUTPUT_DIR}")
    print(f" Tablas guardadas en: {TABLES_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
