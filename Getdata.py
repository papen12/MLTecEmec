import numpy as np
import pandas as pd
import requests

API_KEY = 'cd58317cc56eff29cb8855f727491526'
BASE_URL = 'https://v1.formula-1.api-sports.io/'
headers = {
    'x-rapidapi-host': 'v1.formula-1.api-sports.io', 
    'x-rapidapi-key': API_KEY
}

def cleanData(df):
    df_clean = df.copy()
    
    print("Manejo de valores nulos...")
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna('Desconocido')
    
    print(f"Eliminando duplicados... Encontrados: {df_clean.duplicated().sum()}")
    df_clean = df_clean.drop_duplicates()
    
    df_clean.columns = df_clean.columns.str.strip()  
    df_clean.columns = df_clean.columns.str.lower()  
    df_clean.columns = df_clean.columns.str.replace(' ', '_')  
    
    for col in categorical_cols:
        if col in df_clean.columns:  
            df_clean[col] = df_clean[col].astype(str).str.strip()
            df_clean[col] = df_clean[col].str.replace(r'\s+', ' ', regex=True)  
    
    print("Ajustando tipos de datos...")
    
    date_columns = []
    for col in df_clean.columns:
        if any(keyword in col.lower() for keyword in ['date', 'fecha', 'time', 'hora']):
            date_columns.append(col)
    
    for col in date_columns:
        if col in df_clean.columns:
            try:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                print(f"  Convertida columna '{col}' a datetime")
            except:
                print(f"  No se pudo convertir '{col}' a datetime")
    
    def remove_outliers_iqr(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return series.clip(lower_bound, upper_bound)
    
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = remove_outliers_iqr(df_clean[col])
    
    df_clean = df_clean.reset_index(drop=True)
    
    print("\n" + "="*50)
    print("RESUMEN DE LA LIMPIEZA:")
    print("="*50)
    print(f"Filas originales: {len(df)}")
    print(f"Filas después de limpieza: {len(df_clean)}")
    print(f"Columnas: {df_clean.shape[1]}")
    print("\nTipos de datos:")
    print(df_clean.dtypes.value_counts())
    print("\nValores nulos por columna:")
    print(df_clean.isnull().sum()[df_clean.isnull().sum() > 0])
    
    return df_clean


def get_dataset():
    params = {
        "season": 2023
    }
    
    print("Solicitando datos de carreras de F1 a la API...")
    response = requests.get(BASE_URL + "/races", headers=headers, params=params)

    data = response.json()

    races = []
    for item in data['response']:
        races.append({
            "id": item['id'],
            "competition_name": item['competition']['name'],
            "circuit": item['circuit']['name'],
            "city": item['competition']['location']['city'],
            "country": item['competition']['location']['country'],
            "date": item['date'],
            "season": item['season'],
            "type": item['type'],
            "laps_total": item['laps']['total'],
            "status": item['status']
        })

    df = pd.DataFrame(races)
    print("\nDATA CRUDA:")
    print(df.head())
    
    return df