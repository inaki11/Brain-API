import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def drop_nan_and_nat(df):
    df = df[pd.isna(df['Fecha']) == False]
    df = df[pd.isna(df['Consumo']) == False]
    return df
    
def medidas_por_dia(df):
    medidas = {}
    count = 1
    date = df.iloc[0][0]
    prev = date

    for row in df.values:
        date = row[0]
        if date.day != prev.day or date.month != prev.month:
            medidas[count] = medidas.get(count, 0) + 1
            prev = date
            count = 1
        else:
            count += 1
            
    return medidas


def mostrar_medidas_por_dia(dictionary):
    print(sorted(list(dictionary.items())))
    print("dias con medidas: ", str(sum(list(dictionary.values()))))

def dias_N_medidas(df, value):
    dias_incompletos = set()
    count = 0
    date = df.iloc[0][0]
    prev = date

    for row in df.values:
        date = row[0]
        if date.day != prev.day or date.month != prev.month:
            if count == value:
                dias_incompletos.add((prev.year, prev.month, prev.day))
            prev = date
            count = 1
        else:
            count += 1
    if count == value:
        dias_incompletos.add((prev.year, prev.month, prev.day))
    return dias_incompletos

# Index datetime slicing     -      https://www.dataquest.io/blog/datetime-in-pandas/
def obtener_ventanas_semanas(df):
    win_list = []
    df.set_index('Fecha', inplace=True)
    
    for pos in range(int(len(df)/24) - 6):
        first_date_window = df.index[pos*24]
        date_next_window = df.index[(pos+7)*24 - 1]
        if first_date_window + pd.DateOffset(days=6, hour=23) == date_next_window:
            temp_df = add_velocidad_de_cambio(df[first_date_window : date_next_window].copy())
            win_list.append(temp_df)
            
    return win_list

def add_hora_representacion_ciclica(df):
    df['hora_sin'] = df['Fecha'].map(lambda x: np.sin(x.hour * (2.*np.pi/24)))
    df['hora_cos'] = df['Fecha'].map(lambda x: np.cos(x.hour * (2.*np.pi/24)))
    return df

def add_mes_representacion_ciclica(df):
    df['mes_sin'] = df['Fecha'].map(lambda x: np.sin(x.month * (2.*np.pi/12)))
    df['mes_cos'] = df['Fecha'].map(lambda x: np.cos(x.month * (2.*np.pi/12)))
    return df

def add_velocidad_de_cambio(df):
    prev = df['Consumo'].shift(1)
    prev.iloc[0] = df['Consumo'].iloc[0]
    df['Cambio'] = df['Consumo'] - prev
    return df

def escalar_columnas(df, columnas):
    final_df = df.copy()
    features = df[columnas]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    final_df[columnas] = features
    return final_df