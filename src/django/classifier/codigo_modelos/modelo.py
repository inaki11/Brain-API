from classifier.codigo_modelos.procesado_datos import *
import numpy as np
import tensorflow as tf
import os
from .arquitecturas import * 
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping 
from sklearn.metrics import f1_score


class Modelo:
    """
    Ojo! los prosumidores que no se hayan metido por tener menos datos de una semana seguidos  aparecerán en la lista de entrenamiento pero 
    relamente no habrán sido vistos por el modelo
    """
    def __init__(self, nombre, clases, prosumers, nombre_arquitectura, train_dataset, test_dataset):
        self.nombre = nombre
        self.clases = clases
        self.prosumers = prosumers   
        self.nombre_arquitectura = nombre_arquitectura
        self.nombre_fichero_keras = str(np.random.randint(1, 100000)) + '.keras'
        # modelo_keras carga el modelo keras en memoria. cuando se serializa el modelo entero, se serializa en.keras y se pone a None
        if self.nombre_arquitectura in globals():
        # TO DO,  se necesitan conocer las clases de la BBDD para definir la ultima capa 'softmax' de la arquitectura asi que de momento
        #  se deja hadcodeado = 10 para poder testear
            try:
                input_shape = next(iter(train_dataset.as_numpy_iterator()))[0].shape[1]
                self.modelo_keras = globals()[self.nombre_arquitectura](len(clases), input_shape) #  __init__(default(num_classes, input_shape=1008))
                print(f'modelo creado:   {self.modelo_keras}')
                try:
                    self.modelo_keras.compile(optimizer='adam', loss=CategoricalCrossentropy(), metrics=[CategoricalAccuracy()])
                    print(f'modelo compilado con exito')
                except Exception as error:
                    print(f'Error al compilar:   {error}')
                try:
                    ReduceLR = ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=3, min_lr=0.00001)
                    early_stopping = EarlyStopping(monitor="val_loss", patience = 6)
                    self.modelo_keras.fit(train_dataset, validation_data=test_dataset, epochs=50, callbacks=[ReduceLR, early_stopping])
    
                except Exception as error:
                    print(f'Error al entrenar clasificador:   {error}')

                try:
                    # Make predictions on the test dataset
                    y_pred = self.modelo_keras.predict(test_dataset)
                    # Convert predicted probabilities to class labels
                    y_pred_labels = tf.argmax(y_pred, axis=1).numpy()
                    # Convert one-hot encoded labels to class labels
                    y_true_labels = []

                    for x, y in test_dataset:
                        y_true_labels.extend(tf.argmax(y, axis=1).numpy())

                    # Convert the list to a NumPy array
                    y_true_labels = np.array(y_true_labels)

                    f1 = f1_score(y_true_labels, y_pred_labels)
                    self.f1 = f1
                    print("F1_Score:", f1)
                except Exception as err:
                    print("Error al calcular F1 SCORE" + err)


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
            self.modelo_keras = tf.keras.models.load_model(filepath)
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




