import os
from forecast_demand.codigo_modelos.modelo import Modelo_prediccion
from forecast_demand.codigo_modelos import procesado_datos
import pickle
import json
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import authentication, permissions
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework_simplejwt import views as jwt_views
from drf_spectacular.utils import extend_schema, inline_serializer
from rest_framework import serializers
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import datetime, timedelta
from .codigo_modelos.arquitecturas import *




class detalles_modelo(APIView):
    permission_classes = [permissions.IsAuthenticated]
    @extend_schema(
    request=inline_serializer(
        name="requeste_model_description_forecast_demand",
        fields={
            "prosumer_id": serializers.CharField(),
        },),
    responses=inline_serializer(
        name="response_model_description_forecast_demand",
        fields={
            "prosumer_id": serializers.CharField(),
            "model_architecture": serializers.CharField(),
            "use_weather": serializers.CharField(),
            "prosumer_location": serializers.CharField(),
            "test_scores": serializers.CharField(),  # Por ejemplo
        },),
        )      

    def post(self, request, format=None):
    
        params = request.data
        
        # Seleccionamos el modelo
        if 'prosumer_id' not in params.keys():
            return Response({'response':'Please introduce parameter "prosumer_id" in the request'}, status = status.HTTP_400_BAD_REQUEST)
        
        base_dir =  os.path.dirname(__file__)
        models_path = os.path.join(base_dir, "modelos")
        object_path = os.path.join(models_path, str(params['prosumer_id']) + '.pkl') 
        lista = os.listdir(models_path)

        if params['prosumer_id'] +'.pkl' not in lista:
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
        return Response({"prosumer_name":loaded_classifier_model.nombre,
                         "model_architecture":loaded_classifier_model.nombre_arquitectura,
                         "test_scores":loaded_classifier_model.test_scores,
                         "use_weather":loaded_classifier_model.use_weather,
                         "prosumer_location_code":loaded_classifier_model.prosumer_location}, status = status.HTTP_200_OK)

class eliminar_modelo(APIView):
    permission_classes = [permissions.IsAuthenticated]
    @extend_schema(
    request=inline_serializer(
        name="request_delete_model_forecast_demand",
        fields={
            "prosumer_id": serializers.CharField(),
        },),
    responses=inline_serializer(
        name="response_delete_model_forecast_demand",
        fields={
            "response": serializers.CharField(),
        },),
        )      
    def post(self, request, format=None):
        params = request.data
        if 'prosumer_id' not in params:
            return Response({'response':'The model could not be deleted because a name was not specified as the "prosumer_id" parameter in the request'}, status = status.HTTP_400_BAD_REQUEST)
        # He eliminado el modelo inborrable por defecto ya que en este caso los modelos son exclusivos para cada prosumidor
        try:
            base_dir =  os.path.dirname(__file__)
            models_path = os.path.join(base_dir, "modelos")
            object_path = os.path.join(models_path, str(params['prosumer_id']) + '.pkl')
            lista = os.listdir(models_path)
        except Exception as err:
            print(err)

        if params['prosumer_id'] +'.pkl' not in lista:
            return Response({'response':'There is no model created with that name'}, status = status.HTTP_400_BAD_REQUEST)
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
            
        return Response({'response':'Model ' + str(params['prosumer_id']) + ' successfully deleted'}, status = status.HTTP_200_OK)



class lista_modelos(APIView):
    permission_classes = [permissions.IsAuthenticated]
    @extend_schema(
    responses=inline_serializer(
        name="response_models_list_forecast_demand",
        fields={
            "models_list": serializers.CharField(),
        },),
        )      
    def get(self, request, format=None):
        base_dir =  os.path.dirname(__file__)
        models = os.path.join(base_dir, "modelos")
        lista = os.listdir(models)
        lista.remove('keras_saved_models')
        lista = [x[:-4] for x in lista]
        return Response({'models list':str(lista)}, status = status.HTTP_200_OK)


class ForecastDemandSerializer(serializers.Serializer):
    date = serializers.CharField()
    energy_demand = serializers.ListField()
    is_workday = serializers.BooleanField()

# Tal como está planteado, solo puede existir un modelo por empresa, ya que lleva su nombre. 
# Si se quiere cambiar la arquitectura se deberá llamar a crear uno con el mismo nombre y sobreescribirlo
class crear_modelo(APIView):
    permission_classes = [permissions.IsAuthenticated]         ####TO DO    - >   LLAMADA DE INFERENCIA CONTIENE DATOS HISTORICOS
    @extend_schema(
    request=inline_serializer(
        name="request_create_model_forecast_demand",
        fields={
            "energy_units": serializers.CharField(),
            "prosumer_id": serializers.CharField(),
            "prosumer_name": serializers.CharField(),
            "location": serializers.CharField(),
            "use_weather": serializers.BooleanField(),
            "model_architecture": serializers.CharField(),
            "data": ForecastDemandSerializer(many=True),
        },),
    responses=inline_serializer(
        name="response_create_model_forecast_demand",
        fields={
            "response": serializers.CharField(),
        },),
        )      
    def post(self, request, format=None):
        params = json.loads(request.body)

        if 'energy_units' not in params.keys():
            return Response({'response':'The model could not be created, the "energy_units" parameter is missing in the request body'}, status = status.HTTP_400_BAD_REQUEST)
        if 'prosumer_id' not in params.keys():
            return Response({'response':'The model could not be created, the "prosumer_id" parameter is missing in the request body'}, status = status.HTTP_400_BAD_REQUEST)
        if 'location' not in params.keys():
            return Response({'response':'The model could not be created, the "location" parameter is missing in the request body'}, status = status.HTTP_400_BAD_REQUEST)
        if 'use_weather' not in params.keys():
            return Response({'response':'The model could not be created, the "use_weather" parameter is missing in the request body'}, status = status.HTTP_400_BAD_REQUEST)
        if 'model_architecture' not in params.keys():
            return Response({'response':'The model could not be created, the "model_architecture" parameter is missing in the request body'}, status = status.HTTP_400_BAD_REQUEST)
        if 'data' not in params.keys():
            return Response({'response':'The model could not be created, the "data" parameter is missing in the request body'}, status = status.HTTP_400_BAD_REQUEST)
                
        base_dir =  os.path.dirname(__file__)
        
        if params['model_architecture'] not in globals():
            return Response({'response':'there is not a model architecture named "'+ str(params['model_architecture']) + '" in module "./forecast_demand/codigo_modelos/arquitecturas.py". Please select a valid architecture or include a new one inside that module and create a new docker instance'}, status = status.HTTP_400_BAD_REQUEST)
        
        # Primero creamos el dataset de entrenamiento
        try:
            df, workday_df = procesado_datos.get_df(params['data'])
            df['Value'] = df['Value'].astype('float')
            train_dataset, val_dataset, test_dataset, varianza, media = procesado_datos.process_data(df, params['location'], params['use_weather'] == 'True', workday_df)
        except Exception as err:
            print(err)

        obj_path = os.path.join(base_dir, "modelos", str(params['prosumer_id'])+'.pkl')
        # Creamos el objeto modelo y lo compilamos
        print(f'path del objeto modelo pkl {obj_path}')
        try:
            # Crea la arquitectura del predictivo no es tan trivial, y dependerá la arquitectura de usar o no datos de preciccion históricos, 
            # por tanto es necesario antes de crear el modelo, consultar a la API y crear el Dataset para poder crear la Red Neuronal. Los parámetros necesarios para la 
            # arquitectura por defecto necesita concer tanto el SHAPE de LAS SECUENCIAS DE ENTRADA, como el SHAPE de LAS VARIABLES CONDICIONALES del estado inicial de la RNN
            seq_shape = tuple(next(iter(train_dataset))[0][0].shape[1:].as_list())
            cond_shape = tuple(next(iter(train_dataset))[0][1].shape[1:].as_list())
            input_shapes = (seq_shape, cond_shape)
            print(f"input_shapes: {input_shapes}")
            modelo_1 = Modelo_prediccion(params['prosumer_id'], params['model_architecture'], params['use_weather'], params['location'], input_shapes, (train_dataset, val_dataset, test_dataset), varianza, media) 
        except Exception as err:
            print(err)
        # Lo seleccionamos en global
        loaded_forecast_demand_model = modelo_1
        # Guardamos el modelo 
        print('save model code')
        file = open( obj_path, 'wb')
        modelo_1.save_keras_model()
        pickle.dump(modelo_1, file)    # TO DO en el constructor crea ID aleatoria, nombre del fichero .extension donde guardamos el modelo en /keras_saved_models
        file.close()
        print('guardado pickle')
        loaded_forecast_demand_model.load_keras_model()
        print(f'modelo cargado en memoria contiene en "modelo_keras":  {loaded_forecast_demand_model.modelo_keras}')
        print('--- EXITO ---')
        return Response({'response': f'Electricity consumption prediction model has been successfully created for the prosumer with ID {params["prosumer_id"]}'}, status = status.HTTP_200_OK)                    


class ProsumerSerializerForecastInferenceRequest(serializers.Serializer):
    prosumer_id = serializers.CharField()
    prosumer_name = serializers.CharField()
    user_policies = serializers.ListField()
    next_day_is_workday = serializers.BooleanField()
    next_day_date = serializers.CharField()
    energy_demand = serializers.ListField()
    
class ProsumerSerializerForecastInferenceResponse(serializers.Serializer):
    prosumer_ID = serializers.CharField()
    prosumer_name = serializers.CharField()
    date = serializers.CharField()
    energy_demand = serializers.ListField()

class inferencia(APIView):
    permission_classes = [permissions.IsAuthenticated]
    @extend_schema(
    request=inline_serializer(
        name="request_inference_forecast_demand",
        fields={
            "energy_units": serializers.CharField(),
            "prosumers": ProsumerSerializerForecastInferenceRequest(many=True)
        },),
    responses=inline_serializer(
        name="response_inference_forecast_demand",
        fields={
            "energy_units": serializers.CharField(),
            "prosumers": ProsumerSerializerForecastInferenceResponse(many=True)
        },),
        )
    def post(self, request, format=None):
        response = {}
        #TO DO
        params = json.loads(request.body)
        if 'energy_units' not in params.keys():
            return Response({'response':'The model could not be created, the "energy_units" parameter is missing in the request body'}, status = status.HTTP_400_BAD_REQUEST)
        if 'prosumers' not in params.keys():
            return Response({'response':'The model could not be created, the "prosumers" parameter is missing in the request body'}, status = status.HTTP_400_BAD_REQUEST)
        
        try:
            units = params['energy_units']
            prosumers = params['prosumers']
            for el in prosumers:
                prosumer_id = el["prosumer_id"]
                next_day_date = datetime.strptime(el["next_day_date"], "%d-%m-%Y")
                is_workday = el["next_day_is_workday"] == 'True'
                energy_demand = el["energy_demand"]
                if len(energy_demand) != 24:
                    continue
                
                obj_path = os.path.join(os.path.dirname(__file__), "modelos", prosumer_id +'.pkl')
                with open(obj_path, 'rb') as file:
                    loaded_model = pickle.load(file)
                    loaded_model.load_keras_model()

                # Escalamos los consumos del dia anterior
                energy_demand = (np.array(energy_demand) - loaded_model.media) / loaded_model.varianza  

                # Ectraemos features
                sin_weekday = np.sin(next_day_date.weekday() * (2.*np.pi/6)).reshape(1,1)
                cos_weekday = np.cos(next_day_date.weekday() * (2.*np.pi/6)).reshape(1,1)
                sin_month = np.sin(next_day_date.month - 1 * (2.*np.pi/12)).reshape(1,1)
                cos_month = np.cos(next_day_date.month - 1 * (2.*np.pi/12)).reshape(1,1)
                max_dia_ant = np.max(energy_demand).reshape(1,1)
                min_dia_ant = np.min(energy_demand).reshape(1,1)
                media_dia_ant = np.mean(energy_demand).reshape(1,1)
                workday = np.array([1]).reshape(1,1) if is_workday else np.array([0]).reshape(1,1)
                # Creamos vector de input condicional (de 8 features), que será alimentado a la LSTM 
                input_cond = np.concatenate((sin_weekday, cos_weekday, sin_month, cos_month, max_dia_ant, min_dia_ant, media_dia_ant, workday), axis=0)
                print(f"input cond: {input_cond}")
                # Si el modelo de dicho prosumidor fue entrenado utilizando predicciones, buscamos la predicción de mañana para su localizacion, en api AEMET
                # Añadimos algunos campos de la predicción a las features condicionales
                if loaded_model.use_weather:
                    # Posible error. Si la API KEY se está utilizando para extraer las 6000 predicciones de mañana, es muy probable que nos rechaze la peticion
                    df_aemet = procesado_datos.get_aemet_prediction(loaded_model.prosumer_location)
                    df_aemet = df_aemet[['rain_00-06', 'rain_06-12', 'rain_12-18', 'rain_18-24', 'wind_00-06', 'wind_06-12', 'wind_12-18', 'wind_18-24', 'temp_6', 'temp_12', 'temp_18', 'temp_24', 'hum_6', 'hum_12', 'hum_18', 'hum_24']]
                    print(input_cond.shape, df_aemet.to_numpy().shape)
                    input_cond = np.concatenate([input_cond, df_aemet.to_numpy().reshape(-1,1)]).reshape(1,-1)
                    print(f"cond input shape: {input_cond.shape}")
                energy_demand = energy_demand.reshape(1,-1,1)
                print(f"energy_demand shape: {energy_demand.shape}")
                model_input = [energy_demand, input_cond]
                prediction = loaded_model.modelo_keras.predict(model_input, batch_size=1)
                # Desescalamos la predicción
                print(prediction)
                prediction = (prediction + loaded_model.media) * loaded_model.varianza
                print(prediction)
                response[prosumer_id] = list(prediction)
                # Añadimos prediccion y prosumidor a la respuesta

        except Exception as err:
            print(err)
            
        return Response(response, status = status.HTTP_200_OK)


# TO DO  añadir variables ademas de potencia total a la creacion del dataset

