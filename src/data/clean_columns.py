# src/data/clean_columns.py

import re
import pandas as pd


def normalize_column_name(col: str) -> str:
    """
    Normaliza un nombre de columna a formato limpio y manejable.
    
    Procesos aplicados:
    - strip(): elimina espacios antes y después
    - elimina tabs y saltos de línea
    - convierte espacios a '_'
    - elimina caracteres no alfanuméricos
    - pasa todo a minúsculas (snake_case estándar)
    """

    # Eliminar espacios y caracteres invisibles
    col = col.strip().replace("\t", "").replace("\n", "")

    # Reemplazar espacios por _
    col = col.replace(" ", "_")

    # Eliminar caracteres no permitidos excepto _
    col = re.sub(r"[^A-Za-z0-9_]", "", col)

    # Convertir a minúsculas
    col = col.lower()

    # Evitar dobles guiones bajos
    col = re.sub(r"__+", "_", col)

    return col


def clean_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica la normalización a todas las columnas de un DataFrame.
    """
    df = df.copy()
    df.columns = [normalize_column_name(col) for col in df.columns]
    return df


if __name__ == "__main__":
    # Ejemplo de prueba rápida
    print("Probando limpieza de columnas...\n")

    example_cols = [
        "Curricular units 1st sem approved\t",
        " Admission grade ",
        "Tuition fees up to date",
        "Age at enrollment\n",
        "Marital status:Application Mode"
    ]

    df_test = pd.DataFrame(columns=example_cols)
    df_clean = clean_dataframe_columns(df_test)

    print("Columnas originales:")
    print(example_cols)

    print("\nColumnas limpiadas:")
    print(list(df_clean.columns))
