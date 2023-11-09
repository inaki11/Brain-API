from classifier.codigo_modelos.procesado_datos import *
import numpy as np
import tensorflow as tf
import os
from .arquitecturas import * 
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping 
from cond_rnn import ConditionalRecurrent

"""
    def __init__(self):
        self.nombre = "default_classifier"
        self.clases = ["Oficinas", "Fabrica", "discontinua", "Hospitales", "Hotel", "Supermercado", "Almacen", "Industria_continua"]
        self.empresas = "Empresas obtenidas de datasets publicos debido a la falta de datos durante el desarrollo de los experimentos"
        self.nombre_arquitectura = "default"
        self.nombre_fichero_keras = 'default.keras'
        self.modelo_keras = None
"""

class Modelo_prediccion:
    def __init__(self, nombre, nombre_arquitectura, use_weather, prosumer_location, input_shapes, dataset, varianza, media):
        self.nombre = nombre
        self.nombre_arquitectura = nombre_arquitectura
        self.use_weather = use_weather
        self.prosumer_location = prosumer_location
        self.nombre_fichero_keras = nombre + '.keras'
        self.test_scores = None
        self.varianza = varianza
        self.media = media
        # modelo_keras carga el modelo keras en memoria. cuando se serializa el modelo entero, se serializa en.keras y se pone a None
        if self.nombre_arquitectura in globals():
            try:
                self.modelo_keras = globals()[self.nombre_arquitectura](input_shapes) #  __init__(default(num_classes, input_shape=1008))
                print(f'modelo creado:   {self.modelo_keras}')
                try:
                    self.modelo_keras.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                    print(f'modelo compilado con exito')
                except Exception as error:
                    print(f'Error al compilar:   {error}')

                try:
                    train_dataset, val_dataset, test_dataset = dataset
                    ReduceLR = ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=7, min_lr=0.00001)
                    early_stopping = EarlyStopping(monitor="val_loss", patience = 12)
                    self.modelo_keras.fit(train_dataset, validation_data=val_dataset, epochs=50, callbacks=[ReduceLR, early_stopping])

                    # Evaluamos el modelo y desescalamos
                    self.test_scores = self.modelo_keras.evaluate(test_dataset, verbose=0)
                    """
                    Con el codigo inferior, si se trae la desviacion tipica y la media sustraidas en el Scaler, desde la funcion "process_data"
                    Se podria con esos valores obtener las predicciones desescaladas
                    """

                    #predicciones_escaladas = self.modelo_keras.predict(test_dataset, verbose=0).reshape(-1)
                    #valores_reales_escalados = np.concatenate([y for x, y in test_dataset], axis=0).reshape(-1)
                    #   A continuacion restamos multiplicamos a cada valor por la desviación tipica de consumos y le sumamos la media
                    #valores_reales = valores_reales_escalados*varianza + media
                    #predicciones = predicciones_escaladas*varianza + media

                except Exception as err:
                    print(err)

            except Exception as error:
                print(f'Error al crear el modelo:   {error}')
        else:
            self.modelo_keras = None
            print(f"Se ha creado el modelo sin fichero de keras en '.modelo_keras' ya que la arquitectura solicitada '{self.nombre_arquitectura}' no existe entre los modelos tf.keras definidos en ./arquitecturas.py")

        print("Fin constructor modelo")



    def get_description(self):
        self.summary = ""
        def capture_print(str):
            self.summary += str
        try:
            summary = self.modelo_keras.summary(print_fn=capture_print) # capturamos print para guardarlo en una string y mandarlo en la response
            print(f'string devuelta: -----   {str}')
            return self.summary
        except Exception as err:
            print(err)
        return 
    #f"Model name: {self.nombre} \n Companies trained on: {self.empresas} \n Classes: {self.classes} p\n ML model used: {self.nombre_modelo} \n model description: {self.modelo_keras.summary()}"
    # TO DO   DESCRIPCION ATRIBUTOS DEL MODELO
    
    def load_keras_model(self):
        filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'modelos', 'keras_saved_models', self.nombre_fichero_keras)
        print(f'filepath en load_keras:  {filepath}')
        try:
            self.modelo_keras = tf.keras.models.load_model(filepath, custom_objects={'ConditionalRecurrent':ConditionalRecurrent})
        except Exception as err:
            print(err)
        print(f'loaded model:  {filepath}')
        return

    def save_keras_model(self):
        filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'modelos', 'keras_saved_models', self.nombre_fichero_keras)
        print(f'filepath en save_keras_model:   {filepath}')
        print(f'modelo a guadar:  {self.modelo_keras}')
        try:
            self.modelo_keras.save(filepath, overwrite=True, save_format='keras')
        except Exception as err:
            print(err)

        print(f'modelo "{self.nombre_fichero_keras}" guardado correctamente')
        self.modelo_keras = None
        return

