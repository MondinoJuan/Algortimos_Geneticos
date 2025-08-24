import pandas as pd
import ast

def expand_array_columns(csv_path):
    # Leer CSV
    df = pd.read_csv(csv_path)

    # Detectar columnas con arrays (listas en forma de string)
    array_cols = []
    for col in df.columns:
        if df[col].astype(str).str.startswith("[").any():
            array_cols.append(col)

    print(f"📊 Columnas con arrays detectadas: {array_cols}")

    # Expandir cada columna con arrays
    for col in array_cols:
        # Convertir string a lista real
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x)

        # Determinar tamaño máximo de array en esa columna
        max_len = df[col].dropna().map(len).max()

        print(f"  ↳ Expandiendo {col} en {max_len} columnas...")

        # Crear nuevas columnas
        for i in range(max_len):
            df[f"{col}_{i+1}"] = df[col].apply(lambda x: x[i] if isinstance(x, list) and len(x) > i else None)

        # Eliminar columna original
        df.drop(columns=[col], inplace=True)


    return df


if __name__ == "__main__":
    df = expand_array_columns("df_con_prod.csv")
    output_path = "df_prod_expandido.csv"

    cols = [c for c in df.columns if c not in ["superficie_sembrada_ha", "produccion_tn"]]
    cols += ["superficie_sembrada_ha", "produccion_tn"]

    # Reordenar el DataFrame
    df = df[cols]

    df.to_csv(output_path, index=False)
    print(f"✅ Archivo expandido guardado en: {output_path}")
