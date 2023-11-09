from sklearn.preprocessing import StandardScaler  
import pandas as pd 
import numpy as np
import tensorflow as tf
"""
def hora_representacion_ciclica(arr):
    serie = pd.Series(arr)
    sin = serie.map(lambda x: np.sin(x.hour * (2.*np.pi/24)))
    cos = serie.map(lambda x: np.cos(x.hour * (2.*np.pi/24)))
    return np.concatenate((sin.values.reshape(-1,1),cos.values.reshape(-1,1)), axis=1)

def mes_representacion_ciclica(arr):
    serie = pd.Series(arr)
    sin = serie.map(lambda x: np.sin(x.month * (2.*np.pi/12)))
    cos = serie.map(lambda x: np.cos(x.month * (2.*np.pi/12)))
    return np.concatenate((sin.values.reshape(-1,1),cos.values.reshape(-1,1)), axis=1)

def velocidad_de_cambio(arr):
    prev = np.append(arr[0].reshape(-1,1), arr[0:-1].reshape(-1,1), axis=0)
    return arr-prev
"""
def escalar_columnas(df, columnas):
    final_df = df.copy()
    features = df[columnas]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    final_df[columnas] = features
    return final_df

def unique_prosumer_class(data, ignore):
    # Initialize a set to store unique prosumer classes
    unique_prosumer_classes = set()

    # Initialize a string to store warnings
    warnings = ""

    # Iterate through the prosumers in the JSON data
    for prosumer in data["prosumer"]:
        try:
            prosumer_class = prosumer["prosumer_class"]
            unique_prosumer_classes.add(prosumer_class)
        except KeyError:
            # If prosumer doesn't have a class, add a warning
            ignore.append(prosumer['prosumer_ID'])
            warnings += f"Warning: Prosumer with ID '{prosumer['prosumer_ID']}' dont has 'prosumer_class' key \n"

    return list(unique_prosumer_classes), ignore, warnings

def get_prosumer_class_dict(data):
    prosumer_class_dict = {}

    for prosumer in data['prosumer']:
        try:
            prosumer_class_dict[prosumer['prosumer_ID']] = prosumer['prosumer_class']
        except:
            continue

    return prosumer_class_dict

def get_unique_prosumer_ids_with_warnings(data, ignore):
    # Initialize a set to store unique prosumer IDs
    unique_prosumer_ids = set()
    ignore = []
    # Initialize a string to store warnings
    warnings = ""

    # Iterate through the prosumers in the JSON data
    for prosumer in data["prosumer"]:
        prosumer_id = prosumer["prosumer_ID"]
        if prosumer_id in unique_prosumer_ids:
            # If the ID is repeated, add a warning
            warnings += f"Warning: Prosumer ID '{prosumer_id}' is repeated.\n"
            ignore.append(prosumer_id)
        else:
            # If the ID is not repeated, add it to the set
            unique_prosumer_ids.add(prosumer_id)

    # Convert the set of unique prosumer IDs to a list
    unique_prosumer_ids_list = list(unique_prosumer_ids)

    return unique_prosumer_ids_list, ignore, warnings



def get_dict_dataset(data, ignore, classes):
    """
    Parse and join all data from each prosumer, join it in a df and scale its data using Standard Scaler. 
    Finally add each df to its class list in the dataset dictionary
    """
    dataset = {x:[] for x in classes}
    warnings = ""

    for prosumer in data["prosumer"]:
        prosumer_id = prosumer["prosumer_ID"]
        if prosumer_id in ignore:  # ignore if dont have class or it's id is not unique
            continue
        
        try:
            data = prosumer["data"]
        except:
            warnings += f"the prosumer '{prosumer_id}' dont have any data field, its beeing ignored \n"
            continue

        df = []
        for element in data:
            try:
                date = element["date"] 
                energy_demand = element["energy_demand"]
                if len(energy_demand) != 24:
                    warnings += f"prosumer {prosumer_id} energy array isnt 24 len, date {date} \n"
                    continue
                day = [date] + energy_demand
                df.append(day)

            except Exception as err:
                print(err)
        try:
            df = pd.DataFrame(np.array(df), columns=['date', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24'])  
            df = escalar_columnas(df, columnas=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24'])
            # Columnas que escalar 
            df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y")
            df = df.set_index('date')
            dataset[prosumer['prosumer_class']].append(df)    
        except Exception as err:
            print(err)

    return  dataset, warnings



def generate_7_days_windows(df):
    dataset = []
    date_index = df.index
    for i in range(len(date_index)):
        start_date = date_index[i]
        end_date = start_date + pd.Timedelta(days=6)
        if all(pd.date_range(start_date, end_date, freq='D').isin(date_index)):  # verificamos si existe una secuencia completa de longitud len_seq
            window = pd.date_range(start_date, end_date, freq='D')
            dataset.append(df.loc[window])

    return dataset

def eliminar_clases_pocos_prosumidores(dict_classes_df):
    drop = []
    w = ""
    for key, value in dict_classes_df.items():
        if len(value) < 5:
            drop.append(key)
    
    for key in drop:
        del dict_classes_df[key]
        w += f"Se elimina la clase '{key}' ya que tiene menos de 5 prosumidores \n"
    return dict_classes_df, w

def split_prosumers(dict_classes_df_filtrado):
    class_data_train = {}
    class_data_test = {}

    for key, value in dict_classes_df_filtrado.items():
        l = len(value)
        class_data_train[key] = value[:int(l*0.7)]
        class_data_test[key] = value[int(l*0.7):]

    return class_data_train, class_data_test

def get_slices_per_class(data_dict):
    ret_dict = {key:[] for key in data_dict}
    for key, values in data_dict.items():
        for df in values:
            ret_dict[key] += generate_7_days_windows(df)
    return ret_dict

def print_data_inbalance(train, test):
    print("train")
    for key, value in train.items():
        print(f'data slices key: "{key}": {len(value)}')
    print("test")
    for key, value in test.items():
        print(f'data slices key: "{key}": {len(value)}')


def add_features(dict_slices, dia_semana, mes, velocidad_cambio):
    ret_dict = {key:[] for key in dict_slices}
    for key, slices in dict_slices.items():
        for s in slices:
            aux = s.to_numpy().reshape(1, -1)
            date = s.index[0]
            if velocidad_cambio:
                vel = aux - np.concatenate(([0], aux[:-1]), axis=None)
                aux = np.concatenate((aux, vel), axis=None)
            if dia_semana:
                aux = np.concatenate((aux, np.sin(date.weekday() * (2.*np.pi/6)), np.cos(date.weekday() * (2.*np.pi/6))), axis=None)
            if mes:
                aux = np.concatenate((aux, np.sin(date.month * (2.*np.pi/12)), np.cos(date.month * (2.*np.pi/12))), axis=None)
            ret_dict[key].append(aux)
            
    return ret_dict

def create_tf_dataset(final_dict, classes):
    X = []
    y = []
    num_classes = len(classes)
    for key, values in final_dict.items():
        X += values
        aux = np.zeros(num_classes)
        aux[classes.index(key)] = 1
        y += [aux.copy() for _ in range(len(values))]
    return tf.data.Dataset.from_tensor_slices((X, y))

