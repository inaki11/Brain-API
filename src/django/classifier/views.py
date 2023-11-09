import os
from classifier.codigo_modelos.modelo import Modelo
from classifier.codigo_modelos.heuristicas import get_heuristicas
from .codigo_modelos import procesado_datos
import pickle
import json
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import authentication, permissions
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework_simplejwt import views as jwt_views
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiExample, inline_serializer
from rest_framework import serializers 
from sklearn.preprocessing import StandardScaler 
import tensorflow as tf
import numpy as np
from datetime import datetime, timedelta
import pandas as pd



loaded_classifier_model = None



class detalles_modelo(APIView):
    permission_classes = [permissions.IsAuthenticated]
    @extend_schema(
    request=inline_serializer(
        name="requeste_model_description_classifier",
        fields={
            "model_name": serializers.CharField(),
        },),
    responses=inline_serializer(
        name="response_model_description_classifier",
        fields={
            "model_name": serializers.CharField(),
            "companies_trained_on": serializers.CharField(),
            "classification_classes": serializers.CharField(),
            "model_architecture'": serializers.CharField()
        },),
        )      
    def post(self, request, format=None):
        params = request.data
        
        # Seleccionamos el modelo
        if 'model_name' not in params.keys():
            return Response({'response':'Please introduce parameter "model_name" in the request'}, status = status.HTTP_400_BAD_REQUEST)
        
        base_dir =  os.path.dirname(__file__)
        models_path = os.path.join(base_dir, "modelos")
        object_path = os.path.join(models_path, str(params['model_name']) + '.pkl') 
        lista = os.listdir(models_path)

        if params['model_name'] +'.pkl' not in lista:
            return Response({'response':'The model requested doesnt exist'}, status = status.HTTP_400_BAD_REQUEST)
        else:
            try:
                file_1 = open(object_path, 'rb')
                loaded_classifier_model = pickle.load(file_1)
                loaded_classifier_model.load_keras_model()
                file_1.close()
            except Exception as err:
                print(err)
        ####### 
        return Response({"model_name":loaded_classifier_model.nombre,
                         "classes_trained_on":loaded_classifier_model.clases,
                         "prosumers_trained_on":loaded_classifier_model.prosumers,
                         "model_architecture":loaded_classifier_model.nombre_arquitectura})


class eliminar_modelo(APIView):
    permission_classes = [permissions.IsAuthenticated]
    @extend_schema(
    request=inline_serializer(
        name="request_delete_model_classifier",
        fields={
            "model_name": serializers.CharField(),
        },),
    responses=inline_serializer(
        name="response_delete_model_classifier",
        fields={
            "response": serializers.CharField(),
        },),
        )      
    def post(self, request, format=None):
        params = request.data
        if 'model_name' not in params:
            return Response({'response':'The model could not be deleted because a name was not specified as the "model_name" parameter in the request'}, status = status.HTTP_400_BAD_REQUEST)
        if params['model_name'] == 'default_classifier':
            return Response({'response':'The default model cant be deleted from the API. If you really want to delete it you can do it you should access the docker instance'}, status = status.HTTP_400_BAD_REQUEST)
        base_dir =  os.path.dirname(__file__)
        models_path = os.path.join(base_dir, "modelos")
        object_path = os.path.join(models_path, str(params['model_name']) + '.pkl')
        lista = os.listdir(models_path)
        if params['model_name'] +'.pkl' not in lista:
            return Response({'response':'There is no model created with that model_name'}, status = status.HTTP_400_BAD_REQUEST)
        else: 
            # Primero eliminamos su .keras asociado y despues el objeto wrap
            try: 
                file_1 = open(object_path, 'rb')
                keras_model_filename = pickle.load(file_1).nombre_fichero_keras
                file_1.close()
                os.remove(object_path)
                os.remove(os.path.join(models_path, 'keras_saved_models', keras_model_filename))
            except Exception as err:
                print(err)
                return Response({'response':err})
            
        return Response({'response':'Model ' + str(params['model_name']) + ' successfully deleted'}, status = status.HTTP_200_OK)



class lista_modelos(APIView):
    permission_classes = [permissions.IsAuthenticated]
    @extend_schema(
    responses=inline_serializer(
        name="response_models_list_classifier",
        fields={
            "models_list": serializers.CharField(),
        },),
        )      
    def get(self, request, format=None):
        base_dir =  os.path.dirname(__file__)
        models = os.path.join(base_dir, "modelos")
        lista = os.listdir(models)
        lista.remove('keras_saved_models')
        lista = [x[:-4] for x in lista] # Elimino .pkl para el usuario 
        return Response({'models_list':str(lista)}, status = status.HTTP_200_OK)

# Se intentó con esto pero swagger ignora el segundo nivel
"""
class DataSerializer(serializers.Serializer):
    date = serializers.CharField(),
    energy_demand = serializers.CharField(),
"""

class ProsumerSerializer(serializers.Serializer):
    prosumer_ID = serializers.CharField()
    prosumer_name = serializers.CharField()
    prosumer_class = serializers.CharField()
    data = serializers.ListField()
    # No deja otro serializer encadenado
    #data = DataSerializer(many=True)



# TO DO  ---  METER CLASES Y EMPRESAS
class crear_modelo(APIView):
    permission_classes = [permissions.IsAuthenticated]         ####TO DO    - >   NO SON PARAMETOS SI NO BODY] 
    @extend_schema(
    request=inline_serializer(
            name="request_create_model_classifier",
            fields={
                "model_name": serializers.CharField(),
                "energy_units": serializers.CharField(),
                "model_architecture": serializers.CharField(),
                "prosumer": ProsumerSerializer(many=True),  # Use the nested ProsumerSerializer for the prosumer field
            },),
    responses=inline_serializer(
        name="response_create_model_classifier",
        fields={
            "response": serializers.CharField(),
        },),
        )      
    def post(self, request, format=None):
        params = json.loads(request.body)
        #### CONFIGURACION PRUEBA  ####
        dia_semana = True
        mes = True
        velocidad_cambio = True
        ############
        all_warnings = ""

        if 'model_name' not in params.keys():
            return Response({'response':'The model could not be created, the "model_name" parameter is missing in the request body'}, status = status.HTTP_400_BAD_REQUEST)
        if 'energy_units' not in params.keys():
            return Response({'response':'The model could not be created, the "energy_units" parameter is missing in the request body'}, status = status.HTTP_400_BAD_REQUEST)
        if 'model_architecture' not in params.keys():
            return Response({'response':'The model could not be created, the "model_architecture" parameter is missing in the request body'}, status = status.HTTP_400_BAD_REQUEST)
        if 'prosumer' not in params.keys():
            return Response({'response':'The model could not be created, the "prosumer" parameter is missing in the request body'}, status = status.HTTP_400_BAD_REQUEST)

        # if params['model_architecture'] not in globals():
         #    return Response({'response':f'there is not a model architecture named {params["model_architecture"]} in module "./classifier/codigo_modelos/arquitecturas.py". Please select a valid architecture or include a new one inside that module and recreate the docker instance'}, status = status.HTTP_400_BAD_REQUEST)

        base_dir =  os.path.dirname(__file__)
        obj_path = os.path.join(base_dir, "modelos", str(params['model_name'])+'.pkl')
         # Creamos el objeto modelo y lo compilamos
        print(f'path del objeto modelo pkl {obj_path}')
        
        # SE PUEDE CREAR UN MODELO CLASIFICADOR POR DEFECTO LLAMANDO AL CONSTRUCTOR VACÍO
        try:
            ignore = []
            print("--- 0 ---")
            classes, ignore, warnings = procesado_datos.unique_prosumer_class(params, ignore)
            all_warnings += warnings
            prosumers, ignore, warnings = procesado_datos.get_unique_prosumer_ids_with_warnings(params, ignore)
            all_warnings += warnings
            print("--- 1 ---")
            # Obtenemos dict para mapear prisumer_id a clase
            prosumers = set(prosumers) ^ set(ignore)
            prosumer_class_dict = procesado_datos.get_prosumer_class_dict(params)
            print("--- 2 ---")
            # Parseamos los json de los prosumidores y generamos un df por cada uno. se guardan en una lista asociada a su clase
            dict_classes_df, warnings  = procesado_datos.get_dict_dataset(params, ignore, classes)
            all_warnings += warnings
            print("--- 3 ---")
            # Filtramos clases con menos de 5 prosumidores 
            dict_classes_df_filtrado, warnings = procesado_datos.eliminar_clases_pocos_prosumidores(dict_classes_df)
            all_warnings += warnings
            print("--- 4 ---")
            # recalculamos las clases y prosumidores que entran en el entrenamiento del clasificador
            classes = list(dict_classes_df_filtrado.keys())
            prosumers = [x for x in prosumers if prosumer_class_dict[x] in classes]
            # Necesitamos separar las empresas en train, val y test antes de generar ventanas
            class_data_train, class_data_test = procesado_datos.split_prosumers(dict_classes_df_filtrado)
            #Generamos las ventanas de ambos datasets
            class_data_train = procesado_datos.get_slices_per_class(class_data_train)
            class_data_test = procesado_datos.get_slices_per_class(class_data_test)
            print("--- 5 ---")
            # Prints para ver las cantidad de datos de cada clase
            procesado_datos.print_data_inbalance(class_data_train, class_data_test)
            # Añadimos features dia_semana, mes, velocidad_cambio
            class_data_train = procesado_datos.add_features(class_data_train ,dia_semana, mes, velocidad_cambio)
            class_data_test = procesado_datos.add_features(class_data_test ,dia_semana, mes, velocidad_cambio)
            print("--- 6 ---")
            # Creamos dataset
            train_dataset = procesado_datos.create_tf_dataset(class_data_train, classes)
            test_dataset = procesado_datos.create_tf_dataset(class_data_test, classes)

            train_dataset = train_dataset.batch(16).prefetch(buffer_size=tf.data.AUTOTUNE).cache()
            train_dataset.shuffle(1000)
            test_dataset = test_dataset.batch(16).prefetch(buffer_size=tf.data.AUTOTUNE).cache()
            print("--- 7 ---")
            # Creamos el modelo y lo entrenamos
            modelo_1 = Modelo(params['model_name'], classes, prosumers, params['model_architecture'], train_dataset, test_dataset) 

        except Exception as err:
            print(err)
            return Response({'error': err},  status = status.HTTP_500_INTERNAL_SERVER_ERROR )

        # Guardamos el modelo 
        print('save model code')
        file = open( obj_path, 'wb')
        modelo_1.save_keras_model()
        pickle.dump(modelo_1, file)    # TO DO en el constructor crea ID aleatoria, nombre del fichero .extension donde guardamos el modelo en /keras_saved_models
        file.close()
        print('guardado pickle')
        print('--- EXITO ---')
        return Response({'response':'Model ' + str(params['model_name']) + ' succesfully created'}, status = status.HTTP_200_OK)                    

class classifierInferenceSerializer(serializers.Serializer):
    date = serializers.CharField()
    energy_demand = serializers.ListField()

class inferencia(APIView):
    permission_classes = [permissions.IsAuthenticated]
    @extend_schema(
    request=inline_serializer(
        name="request_inference_classifier",
        fields={
            "model_name": serializers.CharField(),
            "energy_units": serializers.CharField(),
            "prosumer_id": serializers.CharField(),
            "prosumer_name": serializers.CharField(),
            "data": classifierInferenceSerializer(many=True),
        },),
    responses=inline_serializer(
        name="response_inference_classifier",
        fields={
            "prediction": serializers.ListField(),
            "class": serializers.CharField(),
        },),
        )      
    def post(self, request, format=None):

        velocidad_cambio = True
        dia_semana = True
        mes = True

        params = json.loads(request.body)
        base_dir =  os.path.dirname(__file__)
        obj_path = os.path.join(base_dir, "modelos", str(params['model_name'])+'.pkl')

        try:
            model_name = params["model_name"]
            units = params['energy_units']
            data = params['data']
            date = datetime.strptime(data[0]['date'], "%d-%m-%Y")
            consumo = []
            print("__3__")
            # Extraemos consumos del body
            for el in data:
                try:
                    consumo += el['energy_demand']
                except Exception as err:
                    print(err)
            consumo = np.array(consumo)
            # Los escalamos
            consumo = consumo.reshape(7, -1)
            scaler = StandardScaler().fit(consumo)
            consumo = scaler.transform(consumo)
            print("__4__")
            # Añadimos features
            consumo = consumo.reshape(1, -1)
            if velocidad_cambio:
                vel = consumo - np.concatenate(([0], consumo[:-1]), axis=None)
                consumo = np.concatenate((consumo, vel), axis=None)
            if dia_semana:
                consumo = np.concatenate((consumo, np.sin(date.weekday() * (2.*np.pi/6)), np.cos(date.weekday() * (2.*np.pi/6))), axis=None)
            if mes:
                consumo = np.concatenate((consumo, np.sin(date.month * (2.*np.pi/12)), np.cos(date.month * (2.*np.pi/12))), axis=None)

            with open(obj_path, 'rb') as file:
                loaded_model = pickle.load(file)
                loaded_model.load_keras_model()
                print(f"input shape: {consumo.shape}")
                prediction = loaded_model.modelo_keras.predict(consumo.reshape(1,-1), batch_size=1)
                clasification = loaded_model.clases[np.argmax(prediction)]
            print(prediction)
            print(clasification)
            
        except Exception as err:
            print(err)
            return Response({'Error': err}, status = status.HTTP_400_BAD_REQUEST)
        
        return Response({'prediction': str(prediction), 'class': clasification}, status = status.HTTP_200_OK)




class heuristicas(APIView):
    permission_classes = [permissions.IsAuthenticated]
    @extend_schema(
    request=inline_serializer(
        name="request_heuristicas",
        fields={
            "energy_units": serializers.CharField(),
            "data": classifierInferenceSerializer(many=True),
        },),
    responses=inline_serializer(
        name="response_heuristicas",
        fields={
            "consumo_medio_diario_maximo": serializers.CharField(),
            "consumo_horario_medio": serializers.CharField(),
            "consumo_horario_maximo": serializers.CharField(),
            "consumo_horario_minimo": serializers.CharField(),
            "nivel_de_potencia_lunes": serializers.CharField(),
            "nivel_de_potencia_martes": serializers.CharField(),
            "nivel_de_potencia_miercoles": serializers.CharField(),
            "nivel_de_potencia_jueves": serializers.CharField(),
            "nivel_de_potencia_viernes": serializers.CharField(),
            "nivel_de_potencia_sabado": serializers.CharField(),
            "nivel_de_potencia_domingo": serializers.CharField(),
            "desciavion_potencia_lunes": serializers.CharField(),
            "desciavion_potencia_martes": serializers.CharField(),
            "desciavion_potencia_miércoles": serializers.CharField(),
            "desciavion_potencia_jueves": serializers.CharField(),
            "desciavion_potencia_viernes": serializers.CharField(),
            "desciavion_potencia_sabado": serializers.CharField(),
            "desciavion_potencia_domingo": serializers.CharField(),
            "similitud_perfil_consumo_lunes": serializers.CharField(),
            "similitud_perfil_consumo_martes": serializers.CharField(),
            "similitud_perfil_consumo_miercoles": serializers.CharField(),
            "similitud_perfil_consumo_jueves": serializers.CharField(),
            "similitud_perfil_consumo_viernes": serializers.CharField(),
            "similitud_perfil_consumo_sabado": serializers.CharField(),
            "similitud_perfil_consumo_domingo": serializers.CharField(),
            "puntos_de_ruptura": serializers.CharField(),
            "consumo_medio_trabajo": serializers.CharField(),
            "consumo_medio_standby": serializers.CharField(),
            "repetitividad_semanal": serializers.CharField(),
            "repetitividad_laboral": serializers.CharField(),
            "repetitividad_fin_semana": serializers.CharField(),
            "puntos_ruptura_comunes": serializers.CharField(),
            "numero_puntos_laborables": serializers.CharField(),
            "numero_puntos_fin_semana": serializers.CharField()
        },),
        )     
    def post(self, request, format=None):
        params = json.loads(request.body)
        df = []
        try:
            data = params["data"]
            for el in data:
                if len(el['energy_demand']) == 24:
                    df.append([el['date']] + el['energy_demand'])

        except Exception as err:
            print(err)

        try:
            df = pd.DataFrame(np.array(df), columns=['Fecha', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']) 
            df['Fecha'] = pd.to_datetime(df['Fecha'], format="%d-%m-%Y")
            df = df.set_index('Fecha')
            df = np.array(procesado_datos.generate_7_days_windows(df)).astype('int')
            df = df.reshape(df.shape[0], -1)

        except Exception as err:
            print(err)

                    # get_heurísticas
        lista_heuristicas = get_heuristicas(df)

        print(len(lista_heuristicas))

        return Response({'consumo_medio_diario_maximo': str(lista_heuristicas[0]),
                         'consumo_horario_medio': str(lista_heuristicas[1]),
                         'consumo_horario_maximo': str(lista_heuristicas[2]),
                         'consumo_horario_minimo': str(lista_heuristicas[3]),
                         'nivel_de_potencia_lunes': str(lista_heuristicas[4]),
                         'nivel_de_potencia_martes': str(lista_heuristicas[5]),
                         'nivel_de_potencia_miercoles': str(lista_heuristicas[6]),
                         'nivel_de_potencia_jueves': str(lista_heuristicas[7]),
                         'nivel_de_potencia_viernes': str(lista_heuristicas[8]),
                         'nivel_de_potencia_sabado': str(lista_heuristicas[9]),
                         'nivel_de_potencia_domingo': str(lista_heuristicas[10]),
                         'desciavion_potencia_lunes': str(lista_heuristicas[11]),
                         'desciavion_potencia_martes': str(lista_heuristicas[12]),
                         'desciavion_potencia_miércoles': str(lista_heuristicas[13]),
                         'desciavion_potencia_jueves': str(lista_heuristicas[14]),
                         'desciavion_potencia_viernes': str(lista_heuristicas[15]),
                         'desciavion_potencia_sabado': str(lista_heuristicas[16]),
                         'desciavion_potencia_domingo': str(lista_heuristicas[17]),
                         'similitud_perfil_consumo_lunes': str(lista_heuristicas[18]),
                         'similitud_perfil_consumo_martes': str(lista_heuristicas[19]),
                         'similitud_perfil_consumo_miercoles': str(lista_heuristicas[20]),
                         'similitud_perfil_consumo_jueves': str(lista_heuristicas[21]),
                         'similitud_perfil_consumo_viernes': str(lista_heuristicas[22]),
                         'similitud_perfil_consumo_sabado': str(lista_heuristicas[23]),
                         'similitud_perfil_consumo_domingo': str(lista_heuristicas[24]),
                         'puntos_de_ruptura': str(lista_heuristicas[25]),
                         'consumo_medio_trabajo': str(lista_heuristicas[26]),
                         'consumo_medio_standby': str(lista_heuristicas[27]),
                         'repetitividad_semanal': str(lista_heuristicas[28]),
                         'repetitividad_laboral': str(lista_heuristicas[29]),
                         'repetitividad_fin_semana': str(lista_heuristicas[30]),
                         'puntos_ruptura_comunes': str(lista_heuristicas[31]),
                         'numero_puntos_laborables': str(lista_heuristicas[32]),
                         'numero_puntos_fin_semana': str(lista_heuristicas[33])
                         }, status = status.HTTP_200_OK)


# https autofirmado
# separar autenticacion