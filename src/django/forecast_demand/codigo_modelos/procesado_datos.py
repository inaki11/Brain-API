import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ReduceLROnPlateau
import json
import os
import tensorflow as tf
import requests
import time


def process_data(df, localizacion, use_predictions, workday_df):
    # solicitar_historico_consumo

    if use_predictions:
        predictions_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'AEMET', 'predictions', localizacion + '.csv')
        df_aemet = pd.read_csv(predictions_path,  parse_dates=['datetime'], index_col=['datetime'])
        df_aemet.index.rename('Date', inplace=True)
        # TO DO  se podrian incluir  'sky_00-06', 'sky_06-12', 'sky_12-18', 'sky_18-24' pero al ser categoricas no es trivial realizar el one_hot_encodig de la misma forma en entrenamiento y en inferencia
        df_aemet = df_aemet[['rain_00-06', 'rain_06-12', 'rain_12-18', 'rain_18-24', 'wind_00-06', 'wind_06-12', 'wind_12-18', 'wind_18-24', 'temp_6', 'temp_12', 'temp_18', 'temp_24', 'hum_6', 'hum_12', 'hum_18', 'hum_24']]
        

    # Cargar datos historicos temperaturas en un DF, cambiar sus formatos a float y indice por fecha


    # Filtrar features historicos con mas de un 5% de NaN  (Guardar nombres de las columnas para inferencia?)
    # Esto aplicaba cuando se usaban mas colunas de medidas de consumo electtrico, pero ahora mismo solo estoy usando la potencia total porque no me da la vida
    # Si se quiere añadir, simplemente habria que meter las nuevas columnas en "df" y todo funcionaría, ya que el codigo esta adaptado a ello.
    # De hecho probablemente con mas volumenes de datos el rendimiento del mejorará
    nans = df.isna().sum()
    nans = nans / len(df) * 100
    nans = nans.sort_values(ascending=True)
    features = nans[nans<5].index.tolist()
    df = df[features]
    print(f"Features filtradas {features}")
    df.dropna(subset=["Value"], inplace=True)  # Elimini filas con Potencia total vacías

    # Filtramos outliers
    print("Old Shape: ", df.shape)
    Q1 = np.percentile(df['Value'], 25)
    Q3 = np.percentile(df['Value'], 75)
    print(f'Q1:{Q1},  Q2:{Q3}')
    IQR = Q3 - Q1
    upper=Q3+1.5*IQR
    mask = df['Value']<=upper
    df = df[mask]
    print("Upper Bound:",upper)
    #Below Lower bound
    lower=Q1-1.5*IQR
    mask = df['Value']>=lower
    df = df[mask]
    print("lower Bound",lower)
    print("New Shape: ", df.shape)


    # Eliminar dias sin todas las horas
    df['Date'] = df['Date'].dt.floor('H')
    df = df.groupby('Date', as_index=False).mean()
    # elimino dias con horas sin muestras
    df['day'] = df['Date'].dt.floor('D')

    df = df.groupby('day', as_index=False).apply(lambda x: x if len(x)==24 else None)
    df = df.drop(['day'], axis=1)
    index = pd.DatetimeIndex(df['Date'])
    df = df.drop(['Date'], axis=1)

    print("escalado")
    # Escalado de datos   y   Guardo desviacion tipica y media de los datos de consumo total para desescalar las predicciones
    scaler = StandardScaler()
    scaler.fit(df)
    df = scaler.transform(df)

    # Guardo desviacion tipica y media de los datos de consumo total para desescalar las predicciones
    varianza = np.sqrt(scaler.var_)[0]
    media = scaler.mean_[0]
    print(varianza, media)
    print(f'desviación típica: {varianza:.2f}, media: {media:.2f}')

    #  Dataframe poner indice de nuevo tras escalar
    df = pd.DataFrame(df, columns=features[1:])
    df.index = index  


    # --- AÑADIMOS FEATURES ---
    #   Extraemos Features ES_LABORABLE  y la añadimos al dataframe  

    #   SIN COS de Dia de la Semana
    df['dia_sin'] = df.index.map(lambda x: np.sin(x.weekday() * (2.*np.pi/6)))
    df['dia_cos'] = df.index.map(lambda x: np.cos(x.weekday() * (2.*np.pi/6)))

    #   SIN COS de Mes
    df['mes_sin'] = df.index.map(lambda x: np.sin(x.month - 1 * (2.*np.pi/12)))
    df['mes_cos'] = df.index.map(lambda x: np.cos(x.month - 1 * (2.*np.pi/12)))

    #   Variables dia anterior
    arr = np.array(df['Value'])
    arr_media = []
    arr_min = []
    arr_max = []
    for i in range(int(len(arr)/24)):
        for a in range(24):
            arr_media.append(np.mean(arr[i*24:(i+1)*24]))
            arr_min.append(np.min(arr[i*24:(i+1)*24]))
            arr_max.append(np.max(arr[i*24:(i+1)*24]))
        
        # AÑADO 24 0s YA QUE DEBEN ESTAR EN LAS FILAS DEL DIA SIGUIENTE
        # Aunque el dia anterior no siempre es ayer ya que faltan dias en el dataset, no pasa nada ya que al no existir el anterior, no hay X y no se creará ese dia como target.
    df['max_dia_ant'] = np.concatenate([np.zeros(24), np.array(arr_min)[:-24]])
    df['min_dia_ant'] = np.concatenate([np.zeros(24), np.array(arr_max)[:-24]])
    df['media_dia_ant'] = np.concatenate([np.zeros(24), np.array(arr_media)[:-24]])

    # is_workday
    df = pd.merge(df, workday_df, on='Date', how='left')


    if use_predictions:
        #   Temperaturas
        # separo las fechas
        index = df_aemet.index
        features = list(df_aemet.columns)
        # escalo los datos
        scaler = StandardScaler()
        scaler.fit(df_aemet)
        df_aemet = scaler.transform(df_aemet)
        # Dataframe con indice de vuelta
        df_aemet = pd.DataFrame(df_aemet, columns=features)
        df_aemet.index = index
        # Calculamos los datetimes de todas las horas que pertenecen a los dias con predicciones
        all_dates = []
        for date_obj in df_aemet.index:
            hours_of_day = []
            # Create a datetime object with the same date and iterate through hours
            for hour in range(24):
                hour_datetime = datetime(date_obj.year, date_obj.month, date_obj.day, hour, 0)
                hours_of_day.append(hour_datetime)
            all_dates += hours_of_day
        # Eliminamos los dias para los cuales no hay datos históricos de predicciones
        df = df[df['Date'].isin(all_dates)]
        # Agregamos los datos del clima a cada primera hora del dia 00:00, las demas horas del dia quedan rellenas de NaN
        df = df.join(df_aemet, on='Date')
        #display(df)
        
    # Elimino nans de forma temporal de esta forma 
    df = df.fillna(method='ffill').fillna(method='bfill')

    #   Establecer variable (constante)  "variables_condicionales"
    INPUT_DAYS = 1
    variables_condicionales = 8   #  2 X SIN COS de Dia de la Semana     2 X SIN COS de Mes     3 X  Variables dia anterior     1 X Workday

    # Si el modelo usa predicciones históricas, se añaden a la cuenta de features condicionales
    if use_predictions:
        variables_condicionales += df_aemet.shape[1]


    df = df.set_index('Date')
    #display(df)

    #   Creamos ventanas y obtenemos   X_seq, X_condition, y, Date
    #  Esta funcion separa las features del dia siguiente del consumo del dia siguiente, las primeras son X_condition que son parte del input. La segunda el target "y"
    # Finalmente La cantidad de dias de entrada, que tras los experimentos la dejamos como "1", pero se deja el codigo por si se quisiera cambiar en el futuro.
    X_seq, X_condition, y  = crear_ventanas(df, INPUT_DAYS, variables_condicionales)

    #   Dividimos nuestro dataset en solo en Train y Validacion. Creo tupla de val
    train_val_seq, train_val_cond, train_val_y = X_seq[:int(0.8*len(X_seq))], X_condition[:int(0.8*len(X_condition))], y[:int(0.8*len(y))]

    test = (X_seq[int(0.8*len(X_seq)):], X_condition[int(0.8*len(X_condition)):], y[int(0.8*len(y)):])

    # Shufle train y saco validacion de este conjunto
    permutation = np.random.permutation(len(train_val_seq))
    #print(f"permutatioon: {permutation}")
    train_val_seq, train_val_cond, train_val_y = train_val_seq[permutation], train_val_cond[permutation], train_val_y[permutation]

    # Divido train y val
    train_seq, train_cond, train_y = train_val_seq[:int(0.8 * len(train_val_seq))], train_val_cond[:int(0.8 * len(train_val_cond))], train_val_y[:int(0.8 * len(train_val_y))]
    val_seq, val_cond, val_y = train_val_seq[int(0.8 * len(train_val_seq)):], train_val_cond[int(0.8 * len(train_val_cond)):], train_val_y[int(0.8 * len(train_val_y)):]
    #print(f"train_seq: {train_seq.shape}")
    #print(f"val_seq: {val_seq.shape}")
    # Creo tuplas de train y val
    train = (train_seq, train_cond, train_y)
    val = (val_seq, val_cond, val_y)


    #   Se crea tf.Dataset
    train_dataset, val_dataset, test_dataset  = get_dataset(train, val, test)
    return train_dataset, val_dataset, test_dataset, varianza, media


def crear_ventanas(df, INPUT_DAYS, variables_condicionales):
    dates = df.index
    X_seq, X_condition, y = [], [], []

    
    for date in dates.normalize().unique():
        input_start = date
        input_end = date + pd.Timedelta(days=INPUT_DAYS-1, hours=23)  # calculamos la fecha de fin de la secuencia
        target_start = date + pd.Timedelta(days=INPUT_DAYS)
        target_end = target_start + pd.Timedelta(hours=23)
        

        if all(pd.date_range(input_start, target_end, freq='H').isin(dates)):  # verificamos si existe una secuencia completa de longitud len_seq
            input_seq = pd.date_range(input_start, input_end, freq='H')
            target_seq = pd.date_range(target_start, target_end, freq='H')
            
            """
            print("---------------")
            print(input_start, input_end)
            print(target_start, target_end)
            print(input_seq)
            print(target_seq)
            """
            # Si solo hemos cogido la variable consumo total del DF, ajustamos las dimensiones para la LSTM
            if str(type(df)) == "<class 'pandas.core.series.Series'>":
                y.append(np.expand_dims(df.loc[target_seq].to_numpy().astype('float32'), axis=-1))
                X_seq.append(np.expand_dims(df.loc[input_seq].to_numpy().astype('float32'), axis=-1))
                
            # En caso contrario solamente creamos las secuencias
            else:
                y.append(df.loc[target_seq]["Value"].to_numpy().astype('float32'))
                X_seq.append(df.loc[input_seq].to_numpy().astype('float32')[:,:-variables_condicionales])
                X_condition.append(df.loc[target_seq].iloc[0, -variables_condicionales:].to_numpy().astype('float32'))
                """
                print(input_seq)
                print(target_seq)
                display(X_seq[-1])
                display(X_condition[-1])
                display(y[-1])
                print('-------------')
                """
                
    return np.array(X_seq), np.array(X_condition), np.array(y)


def get_dataset(train_fold, val_fold, test_fold):
    X_seq_train, X_condition_train, y_train = train_fold
    X_seq_val, X_condition_val, y_val = val_fold
    X_seq_test, X_condition_test, y_test = test_fold
    
    train_input = tf.data.Dataset.from_tensor_slices((X_seq_train, X_condition_train))
    train_target =  tf.data.Dataset.from_tensor_slices(y_train)
    train =  tf.data.Dataset.zip((train_input, train_target))

    val_input = tf.data.Dataset.from_tensor_slices((X_seq_val, X_condition_val))
    val_target = tf.data.Dataset.from_tensor_slices(y_val)
    val = tf.data.Dataset.zip((val_input, val_target))

    test_input = tf.data.Dataset.from_tensor_slices((X_seq_test, X_condition_test))
    test_target = tf.data.Dataset.from_tensor_slices(y_test)
    test = tf.data.Dataset.zip((test_input, test_target))

    train = train.batch(64).prefetch(buffer_size=tf.data.AUTOTUNE).cache()
    train.shuffle(1000)
    val = val.batch(64).prefetch(buffer_size=tf.data.AUTOTUNE).cache()
    val.shuffle(1000)
    test = test.batch(64).prefetch(buffer_size=tf.data.AUTOTUNE).cache()

    #print(y_train.shape)
    print(f'Train shapes:   Condition:{X_condition_train[0].shape}  Seq:{X_seq_train[0].shape}  target:{y_train[0].shape}')
    print(f'Val shapes:   Condition:{X_condition_val[0].shape}  Seq:{X_seq_val[0].shape}  target:{y_val[0].shape}')
    print(f'test shapes:   Condition:{X_condition_test[0].shape}  Seq:{X_seq_test[0].shape}  target:{y_test[0].shape}')
   
    return train, val, test


def get_df(data):
    df = []
    workday_df = {'Date':[], 'Workday':[]}

    for el in data:
        try:
            date = el['date']
            date_obj = datetime.strptime(date, "%d-%m-%Y")
            # Initialize the list to store all hours
            hours_of_day = []

            # Create a datetime object with the same date and iterate through hours
            for hour in range(24):
                hour_datetime = datetime(date_obj.year, date_obj.month, date_obj.day, hour, 0)
                hours_of_day.append(hour_datetime)

            energy = el['energy_demand']
            # Create energy demand array
            if len(energy) != 24:
                continue
            
            # Create disct with working days
            workday_df['Date'] += [date_obj]
            workday_df['Workday'] += [1] if el['is_workday'] else [0]


            arr = np.column_stack([hours_of_day, energy])
            df.append(arr)
        except Exception as err:
            print(err)

    return pd.DataFrame(np.concatenate(df, axis=0), columns=['Date', 'Value']), pd.DataFrame.from_dict(workday_df)


def get_aemet_prediction(id):
    API_key_aemet = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJzaWx2aWEuZ29uemFsZXouaXRjbEBnbWFpbC5jb20iLCJqdGkiOiIzYmEwZDM2Yy1mMzVjLTRhN2ItYTNlMi0zYTA2Y2RjNTlkYzUiLCJpc3MiOiJBRU1FVCIsImlhdCI6MTUzNTA5MzA3NywidXNlcklkIjoiM2JhMGQzNmMtZjM1Yy00YTdiLWEzZTItM2EwNmNkYzU5ZGM1Iiwicm9sZSI6IiJ9.8unUtDB9bkmmGbUN7CfJIXZZzMyhH7t6242zRA2WbUw"
    #time.sleep(2.5)
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
            return pd.DataFrame(df_dict, index=[0]).set_index(['datetime'])
        else:
            print("la prediccion para mañana no está en la posicion 1 de la respuesta de AEMET")
    else:
        print("Status respuesta es distinto de 200 !!!")
    return