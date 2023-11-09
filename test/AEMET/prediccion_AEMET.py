import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime, timedelta
import json
import time


base_path = os.path.dirname(os.path.abspath(__file__))

df_municipios = pd.read_csv(base_path + '/id_municipios.csv', dtype={'NOMBRE':str, 'ID':str}, )
df_municipios = df_municipios.dropna()


API_key_aemet = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJzaWx2aWEuZ29uemFsZXouaXRjbEBnbWFpbC5jb20iLCJqdGkiOiIzYmEwZDM2Yy1mMzVjLTRhN2ItYTNlMi0zYTA2Y2RjNTlkYzUiLCJpc3MiOiJBRU1FVCIsImlhdCI6MTUzNTA5MzA3NywidXNlcklkIjoiM2JhMGQzNmMtZjM1Yy00YTdiLWEzZTItM2EwNmNkYzU5ZGM1Iiwicm9sZSI6IiJ9.8unUtDB9bkmmGbUN7CfJIXZZzMyhH7t6242zRA2WbUw"



list_dir = os.listdir(base_path + '/predictions')

for id in df_municipios['ID']:
    try:
        time.sleep(2.5)

        url_horaria = f"https://opendata.aemet.es/opendata/api/prediccion/especifica/municipio/diaria/{id}"
        # Realizar la solicitud GET a la API
        response = requests.get(url_horaria, headers={"Authorization": "Bearer " + API_key_aemet})

        # Comprobar si la solicitud fue exitosa
        if response.status_code == 200:
            # Obtener la URL de los datos
            data_url = response.json()["datos"]
                    # Hacer una segunda solicitud GET a la URL de los datos
            data_response = requests.get(data_url)

            # transformar a json
            data_json = data_response.json()

            # Obtengo fecha de mañana
            tomorrow_date = pd.to_datetime(datetime.today() + timedelta(1)).floor('d')
    
            
            if  pd.to_datetime(data_json[0]['prediccion']['dia'][1]['fecha']) == tomorrow_date:
                data= data_json[0]['prediccion']['dia'][1]

                df_dict = {
                    **{'datetime': pd.to_datetime(data['fecha'])},
                    **{'rain_' + x['periodo']:x['value'] for x in data['probPrecipitacion']},
                    **{'sky_' + x['periodo']:x['descripcion'] for x in data['estadoCielo']},
                    **{'wind_' + x['periodo']:x['velocidad'] for x in data['viento']},
                    **{'temp_' + str(x['hora']):x['value'] for x in data['temperatura']['dato']},
                    **{'temp_max': data['temperatura']['maxima'], 'temp_min':data['temperatura']['minima']},
                    **{'hum_' + str(x['hora']):x['value'] for x in data['humedadRelativa']['dato']}
                }
                df_row = pd.DataFrame(df_dict, index=[0]).set_index(['datetime'])

                if id + '.csv' not in list_dir:
                
                    df_row.to_csv(f'{base_path}/predictions/{id}.csv')
                    print('New csv created: ')

                else:
                    #print('Reading csv...')
                    df = pd.read_csv(f'{base_path}/predictions/{id}.csv',  parse_dates=['datetime'], index_col=['datetime'])
                    #print('After add row if datetime not already in')
                    if tomorrow_date not in df.index:
                        df = pd.concat([df, df_row], ignore_index=False)
                        df.to_csv(f'{base_path}/predictions/{id}.csv')
                        print('Added new row')
                    else:
                        print('Already find a prediction for this date')


            else:
                with open(base_path + '/errores.txt', 'a') as f:
                    f.write(f'{datetime.now()}  id: {id}  *** No coincide la posicion 1 con la fecha de mañana \n')
                print('*** No coincide la posicion 1 con la fecha de mañana')
                

        else:
            with open(base_path + '/errores.txt', 'a') as f:
                f.write(f'{datetime.now()}  id: {id}  Response: {response} \n')
            print(response)

    except Exception as err:
        print(f'exceptcion en id: {id},  excepcion: {err}')
        with open(base_path + '/errores.txt', 'a') as f:
            f.write(f'{datetime.now()}  id: {id}  Excepcion: {err} \n')
            
with open(base_path + '/exito.txt', 'a') as f:
    f.write(f'{datetime.now()}  He acabado de ejecutar todas las filas \n')

