# Documentación API Brain

## Indice
1. Introducción al proyecto
2. Experimentos
    1. Introducción
    2. Experimentos Clasificador
    3. Experimentos Predictivo Consumo Eléctrico
3. La API
    1. Introducción
    2. Dockerizado, contenedores Nginx y Django
    3. Contenedor django en profundidad
        1. Intuición general de django
        2. App AEMET
        3. App classifier
        4. App forecast_demand
4. Como usar la API
    1. Despliegue local sin Docker
    2. Despliege con Docker

## Introducción al proyecto
Este repositorio contiene mi parte del proyecto de Brain. Esta parte consiste en la creación de una API que permita crear y utilizar modelos de clasificación/segmentacion de prosumidores a partir de su perfil de consumo eléctrico, así como de predicción de consumo eléctrico y generación fotovoltaica. 

Sin embargo, solo ha sido desarrollado hasta este punto la API y el código relacionado con la comunicación, modelos de clasificacion y predicción de consumo. Quedando por tanto pendiente tanto la experimentación de modelos de predicción de generación fotovoltaica como la implementacion de la parte de la API encargada de crear los modelos y responder a las peticiones de predicciones de generación de los prosumidores.

Posteriormente, esta API de predicción tendrá que enviar las predicciones al módulo de ajuste que desarrollarán los chicos de energía. Pasandole con estas predicciones unos valores de política de usuario. Finalmente ese módulo deberá enviar las respuestas al cliente. Toda esta parte de momento está en el aire, y pendiente de desarrollo.

A continuacion se muestra un gráfico con las diferentes partes a implementar, aunque esto es todavía muy susceptible de cambios y se debate dia a dia en las reuniones de arquitectura.

![flujo](/readme_images/flujo.png)

## Experimentos

#### Introducción
La experimentación realizada queda documentada en la carpeta **/test**. Aqui podemos encontrar en dos directorios separados la experimentación para el clasificador y la del predictivo.
Además también se encuentra en esta carpeta el código para la recoleccion de predicciones de AEMET y su almacenamiento.

#### Experimentos Clasificador
El código se divide en dos jupyter notebook, **"Brain_Clasificador"** y **"Heurísticas"**.

* **Brain_Clasificador** es el primer código realizado por mi parte en este proyecto, y se trata de intentar implementar diferentes algoritmos de Machine Learning para clasificar espresas por su tipo de consumo eléctrico.
El alcance de esta parte estuvo muy limitada ya que no teníamos casi datos por parte del cliente, apenas 4 empresas distintas. Por ello se optó por probar a descargar datos publicos en internet. Se probaron clasificadores como random forest y redes neuronales.

* **Heurísticas** es la implementación en python del código desarrollado para la caracterización de perfiles de consumo, por parte de el departamento de energía. Respecto a las dudas con esto, se deberá consultar a Aaron, que fue quien realizó esta "lógica", hasta donde yo se.

#### Experimentos Predictivo Consumo Eléctrico
En la carpeta de experimentacion se pueden encontrar los siguientes ficheros: 

* **Pruebas Predictivo.ipynb**  - Es el Notebook usado para la experimentacion de features y arquitecturas para el modelo.
En este jupyter se centra la mayoría de mi trabajo realizado, siendo donde se han realizado tanto el procesamiento de los datos, extracción y seleccion de características, pruebas de arquitecturas e hiperparámetros, validacion cruzada y análisis de los resultados. 
Toda la experimentacion se realizan sobre "Datos_v7", que son datos de consumo de CPS_Getafe proporcionados por Aaron.
* **Datos_v7.RData**  -   Son los datos históricos de consumo de CPS_Getafe
* **dataset_inputs.csv**  -  Este csv sirve para almacenar de forma persistente las diferentes pruebas realizadas manualmente para la búsqueda de features para el modelo. Almacena que variables se incluyeron, asi como técnicas relaciondas con la elaboracion  del dataset (como oversamplig), junto con el rendimiento dado en el testeo.
* **observaciones_clima_Getafe.json**  -  Dado que no se puede solicitar a traves de la API de AEMET las predicciones pasadas, se quiso probar con la observaciones reales pasadas. Aunque resultaron pobres tambien, se decidió elaborar un historico de predicciones de ahora en adelante para probar su efectividad en el futuro, ya que además se prevee su uso en los modelos de predicción de generacion fotovoltaica
* **best_hps**  -  Almacenamiento persistente de los experimentos automáticos de búsqueda de hiperparámetros y arquitectura, mediante la librería "keras_tuner"

---
## La API

### Introducción
La API es el servicio dockerizado que realmente se va a utilizar, y que permite la creacion de modelos y su uso mediante request https con un body de datos en ".json"
El código de la API se encuentra en **"/src"**

### Dockerizado, contenedores Nginx y Django
Cuando accedemos a **/src** vemos un fichero llamado **"docker-compose"** y dos directorios, **"django"** y **"nginx"**

Docker compose es un ficehero de configuración de docker, que sirve para definir como ejecutar un servicio "compuesto" por mas de un contenedor que trabajan juntos.

En nuestro proyecto, de momento, solo hay **dos contenedores "django" y "nginx"**, que estan definidos cada uno en una carpeta. **django** es un contenedor que tiene toda la lógica de la API, es decir todos los endpoints y el codigo para crear datasets, modelos y generar la inferencia. Finalmente, **nginx** es el contenedor que se encarga de recibir las solicitudes https, desencriptarlas y redirigirselo al contenedor "docker", asi como recibe las respuestas http de docker las encripta antes de devolver la respuesta al cliente.

Esta encriptacion depende de un **certificado autofirmado** que se halla en /ngix, y quizas seria responsable generar otro nuevo al desplegarlo en producción.

Dentro de cada una de estas carpetas, existe un fichero de configuracion llamado "Dockerfile" que son las instrucciones para contruir cada uno de los contenedores.

![Dockerfile de django](/readme_images/Dockerfile_django.PNG)
![Dockerfile de nginx](/readme_images/Dockerfile_nginx.PNG)
![docker-compose](/readme_images/docker-compose.PNG)


---
### Contenedor django en profundidad

#### Intuición general de django
El servidor de django se estructura de la siguiente forma:
* **servidor_brain**  - Es el directorio fundamental del servidor. En el están el fichero de configuración "settings.py" y el fichero para la definicion de los endpoints "urls.py" entre otros. En la práctica en principio a partir de como está el desarrollo no debería ahcer falta modificar ninguno mas.
* **AEMET** - Es la carpeta donde se encuentra la lógica para solicitar predicciones y almacenarlas, mas los datos históricos de predicciones. Se verá con detalle en el siguiente punto. Destacar que **No es una app de servidor** ya que no tiene endpoints, es decir no recibe peticiones ni da respuestas, es de uso interno del servidor.
* **classifier y forecast_demand** - Son apps del servidor.
tienen sus propios endpoints, separando asi visualmente la lógica de los servicios.
Por ejemplo **/forecast_demand/create_model vs  /classifier/create_model**.
En principio dentro de cada app solo prestaremos atencion a **"views.py"**, que contiene las funciones que ejecutan los endpoints (crear modelo, borrar modelo, inferencia...), **"urls.py"** que mapea esas funciones con los endpoints.
Para que views.py pueda hacer toda la lógica de la API se apoya en dos subdirectorios de la app: **"codigo_modelos" y "modelos"**.  
    * **codigo_modelos**    contiene:
        * **arquitecturas.py** - Es donde se definen las arquitecturas de IA para crear los modelos. Al crear el modelo se pasa por json el parámetro "model_architecture". Este valor tiene que coincidir con alguna clase definida en este módulo. Si coincide se creará un modelo con esa arquitectura.
        * **modelo.py** - Aquí se define la clase del modelo que se crea. Esta tiene como atributo el modelo de tensorflow asociado, pero tiene a su vez varios atributos que dan informacion de modelo creado. Estos atributos, y esta clase, como cabe esperar son muy distintos en las diferentes apps pero siguen la misma lógica. 
        * **procesado_datos.py** es un módulo donde se guardan todas las funciones que se necesitna para extraer los datos dado del json, procesarlos, escalarlos, añadir features (por ejemplo aññadir predcciones de clima) y crear los conjuntos de entrenamiento, validacion y test.
        * **heuristicas** este es un caso especial y solo se da en el clasificador. contiene la logica de la caracterizacion por heurísticas desarrollada por los chicos de enrgía.
    * **modelos** contiene:
        * Todos los objetos modelo serializados que han sido creado en la app **nombre_modelo.pkl**
        * subdirectorio **keras_models** que contiene los modelos keras serializados en .h5 (cada vez que se carga un objeto modelo se debe llamar a su método "load_keras_model" para que deserialize y cargue en su atributo el modelo con sus pesos)

![django_contenedor](/readme_images/django_contenedor.png)

---
#### Carpeta AEMET
En la app de AEMET se encuentran 3 elementos. 
1. El primero es un "id_municipios.csv", el cual mapea cada uno de los municipios de España con un código identificador propio de la AEMET
1. El segundo es un directorio llamado "predictions" que contiene un csv por cada municipio de España. Siendo su codigo en el documento id_municipios.csv, el nombre de cada fichero csv.
(Estas Prediciones son utilizadas para crear los datasets de entrenamiento de los modelos de predicción de consumo, y se prevee que quizas se usen en los modelos de predicción de generación fotovoltaica. Solo son utilizados si se indica "use_weather" = True en el json de solicitud de creación del modelo. En caso ser True los dias que no tengan historico de prediccion almacenado son eliminados.)
1. El tercero y último es un script de python "prediccion_AEMET.py" que si se ejecuta realiza una petición de prevision meteorológica a la AEMET por cada uno de los municipios de España. Por cada municipio, extrae ciertos datos de interés de la predicción climatológica para el día siguiente, y añade una fila al fichero csv con el codigo de dicho municio.
Es decir, el script lo que hace es consultar las previsiones para mañanas y almacenar 6000 filas en 6000 csv distintos.
El principal problema de esto es que la AEMET limita mucho las solicitudes, siendo necesario espaciar las request varios segundos, por lo que la ejecucion dura una 7 horas. Actuamente está desplegado este script y la recogida de datos mediante un cronjob en el servidor interno de jupyter https://192.168.8.30:8000/ en el directorio  inaki.campo/Brain_AEMET

QUEDA PENDIENTE AUTOMATIZAR LA RECOGIDA Y ALMACENAMIENTO DE LOS DATOS DENTRO DEL SERVICIO DOCKER. 
Se recomienda que en el momento de realizar esto, se traigan los datos csv de predicciones del servidor para que los datos recogidos hasta ahora no se pierdan y se tenga un mayor conjunto de datos de predicciones.

PARA ELLO HAY QUE AÑADIR UN CRONJOB QUE EJECUTE ESTE FICHERO CADA DIA A MEDIA MAÑANA (10 AM APROX)

**Recordar:** La carpeta "django" es en si un "volumen" de docker, es decir, una carpeta compartida entre nuestra máquina y el contenedor. Es decir, cualquier cambio que realize una instancia de docker sobre ese directorio o sus subdirectorios será persistente. Por tanto, dado que las "apps" AEMET, forecast_demand, y classifier estan contenisdas en subdirectorios de este volumen, cualquier modelo que se crea por llamadas a la API o recodidas de datos meteorológicos, son persistentes si se cae el docker y se crea una nueva instancia.

---
#### App classifier
Recordamos, la estructura de la app **classifier**  es la siguiente:


* **codigo_modelos**    contiene:
    * **arquitecturas.py**  
    * **modelo.py**
    * **procesado_datos.py**
    * **heuristicas** 
* **modelos** contiene:
    * Todos los objetos modelo serializados que han sido creado en la app **nombre_modelo.pkl**
    * subdirectorio **keras_models**
* **views.py** - Esto es el núcleo, y es donde se implementan los endpontis de la API
    * class detalles_modelo
    * class eliminar_modelo
    * class lista_modelos
    * class crear_modelo
    * class inferencia
    * class heuristicas
        
Procedemos a describir el el código de cada uno de ellos:


* **arquitecturas.py:**
```python
def default(num_classes, input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))
    return model
```
La arquitectura 'default' de clasificacion es muy simple y esta formada por dos capas ocultas de 32 y 8 neuronas. Cabe destacar que la falta de datos impidió una experimentacion mayor y la búsqueda de una arquitectura mas sofisticada. Es por ello que se deja la API lista para añadir aqui una o mas nuevas clases de arquitectura. Para hacer uso simplemente habrá que pasar el nombre de la clase por parámetero, al llamar a create_model. (parámetro "model_architecture")

Indicar finalmente que en caso de añadir una nueva clase, hay que tener cuidado de que no sea un nombre ya utilizado previamente para nombrar otra clase, ya que la búsqueda de ésta es realizada buscando en "globals", en la creacion del modelo (en el init de modelo.py)

* **modelo.py:** Consta de las siguiente partes:

    * __init__ Construye, entrena y evalua el modelo con el dataset y el nombre de arquitectura dado. 
                    
        * EL constructor __init__ que toma varios argumentos: nombre (nombre del modelo), clases, prosumers, nombre_arquitectura, train_dataset y test_dataset.
        * La función asigna estos argumentos a atributos de la instancia de la clase.
        * Se genera un nombre aleatorio para el fichero .keras y se guarda como atributo
        * Se busca la clase con  el nombre de la arquitectura dada, y se instancia un objeto modelo
        * Se compila con Adam y CategoricalCrossentropy
        * Se entrena ,monitorizando la pérdida en el conjunto de validacion, y se hace uso de los callbacks **ReduceLROnPlateau**(monitor="val_loss", factor=0.7, patience=3, min_lr=0.00001) **EarlyStopping**(monitor="val_loss", patience = 6), que reduce el Learning Rate si no se mejora en 3 epoch, y se para el entrenamiento a a las 6 epocas sin mejora. LOS VALORES DE LOS CALLBACKS SON DE PRUEBA Y SE DEBERÁN AJUSTAR EN EL FUTURO
        * Finalmente se evalua el modelo clasificador con el conjunto de test con la métrica **F1 score**, que se guarda en "self.f1 "
    

    * **load_keras_model** - Se encarga de cargar el modelo keras serializado y cargarlo en el atributo "self.modelo_keras"

    * **save_keras_model** - Guarda el modelo entrenado serializado en el directorio modelos, con el nombre del fichero generado que está guardado en el atributo "self.nombre_fichero_keras"

* **procesado_datos.py** - Es un modulo con funciones para la creacion de los **datasets** a partir del json recibido en **create_model**
Clases:
    * **unique_prosumer_class** - input json crudo, hace lista de las clases de los prosumidores dados para crear el modelo. devuelve tambien una lista con id's de los prosumidores dados que no tienen clase en el json dado, con el fin de usar la lsita para descartarlos.
    * **get_unique_prosumer_ids_with_warnings** - Recibe el json crudo y comprueba si hay prosumidores duplicados, en caso de que si los añade a la lista de ignorar. Devuelve la lista de los prosumidores válidos.
    * **get_prosumer_class_dict** - crea un diccionario auxiliar que contiene en cada clave una id de prosumidor y en el valor su clase dada.
    * **get_dict_dataset** - Recorre todos los prosumidores dados válidos, por cada uno de ellos recorre todos los días de datos dados, coge los 24 datos de 24h de consumo y la fecha y crea un dataframe que en cada fila tiene 24 valores de consumo, y como indice un datetime con la fecha.  Finalmente, crea un diccionario cuyas claves son todas las clases dadas, y el valor es una lista con los dataframes de todos los prosumidores válidos con dicha clase asignada.   Es decir *{"class_1":[df_prosumer_1, df_prosumer_3], "class_2":[df_prosumer_2, df_prosumer_4], "class_3":[df_prosumer_11, df_prosumer_32] ...}*
    * **eliminar_clases_pocos_prosumidores** - Eliminas las clases que contengan menos de N prosumidores en los datos de entrenamiento dados. En principio he puesto como valor 5.  *Esto es necesario ya que se necesitan varios prosumidores distintos ya que no se piueden mezclar sus datos en train, validacion y test.*
    * **split_prosumers** - Separa en dos diccionarios Train y test  (no puse validacion ya que considero que el test real será en la práctica y estoy ajustando sobre test.  He decidido a ojo dividir 70/30)
    * **get_slices_per_class** - Ya que necesita el clasificador que se netrenó 7 dias contiguos, se obtienen de cada DF de cada clase, todas las ventanas de 7 dias contiguos y se almacenan en un diccionario tipo *{'clase_1':[ventana_1_df, ventana2_df....], 'clase_2':[ventana_1_df, ventana2_df....]}*
    * **add_features** - A cada ventana se le añaden: *codificacion seno coseno de dia_semana, mes y la cantidad de cambio respecto a la hora anterior* (Estas features son cuestionables como el modelo clsificador en general, pero es lo que se hizo en experimentación.)
    * **create_tf_dataset** - Se transforma cada uno de estos diccionarios a objetos dataset de tensorflow. con las lables transformadas de las clases validas a una codificacion one-hot.

* **views.py** - Ejemplo estructura de un endpoint:
Una imagen vales mas que muchas palabras :)
![views_ejemplo_estructura](/readme_images/views_ejemplo_estructura.png)
---

#### App forecast_demand
Recordamos, la estructura de la app **forecast_demand** es la siguiente:
 
* **codigo_modelos**    contiene:
    * **arquitecturas.py**
    * **modelo.py**
    * **procesado_datos.py**
* **modelos** contiene:
    * Todos los objetos modelo serializados que han sido creado en la app **nombre_modelo.pkl**
    * subdirectorio **keras_models**
* **views.py** - Esto es el núcleo, y es donde se implementan los endpontis de la API
    * class detalles_modelo
    * class eliminar_modelo
    * class lista_modelos
    * class crear_modelo
    * class inferencia


Procedemos a describir el el código de cada uno de ellos:

* **arquitecturas.py:**
```python
def default_demand_forecast(parametros_arquitectura):
    input_1_shape, variables_condicionales_shape = parametros_arquitectura
    input1 = Input(shape=input_1_shape)
    input2 = Input(shape=variables_condicionales_shape)
    x = Conv1D(filters=1200, kernel_size=3, strides=3, padding="causal", activation="relu", input_shape=input_1_shape)(input1)
    x = ConditionalRecurrent(LSTM( 900, input_shape=input_1_shape, activation='tanh', return_sequences=True))([x, input2])
    x = LSTM(1100, activation='tanh')(x)
    x = Dropout(0.1)(x)
    output = Dense(24)(x)
    model = Model(inputs=[input1, input2], outputs=output)
    return model
```
**default_demand_forecast** es la arquitectura seleccionada en la experimentación para la predicción de consumo eléctrico. Debido a la poca cantidad de datos de una sola empresa que disponiamos (un año y con muchas lagunas), es posible que en el futuro esta misma arquitectura pero añadiendo algo mas de complejidad (mas neuronas de LSTM y filtros), se comporte mejor. Esto es ya que por la falta de datos hemos sufrido mucho el "sobreajuste" de los modelos, y para tratar de contener este problema nos hemos visto obligados a hacer modelos mas pequeños. Sin embargo con mas datos quizas este problema se solucione y se puedan extraer patrones mas complejos y mejorar el comportamiento de los modelos.

**Explicacion detallada de la arquitectura escogida**

La arquitectura consta de una primera capa convolucional de una dimension que recibe como input solo los datos de consumo electrico del dia anterior.  

La segunda capa es sensiblemente mas compleja. Ésta es una capa LSTM especial "**ConditionalRecurrent**" que hemos cogido prestada de la librería cond_nn.  Esta capa LSTM que coge como valor de entrada secuencial los datos de consumo del dia anterior pasados por los filtros, como una LSTM normal, pero la particularidad es que el se inicial la LSTM con un estado inicial fruto del aprendizaje de un input "condicional" (Es decir, que condiciona la salida. En nuestro caso por ejemplo si el dia siguiente es festivo, que dia de la semana es, o la prediccion de clima).Finalmente esta LSTM se configura como "return sequences" ya que vamos a utilizarla debajo de otra LSTM para añadir complejidad a los patrones que puede extraer.

![cond_rnn](/readme_images/cond_rnn.png)

La tercera capa es una LSTM convencional, que como se ha comentado, recibe la secuencia formada por cada una de lasetapas de la LSTM anterior.

La cuarta capa añade un poco de dropout para intentar lidiar con el sobreajuste

La quinta y ultima capa son 24 neuronas que reciben todo el conocimiento generado por las capas entariores, y tratan de estimar el consumo de las proximas 24h

* **procesado_datos.py** - Este módulo contiene las funciones para extrae los datos dado en el json y crear el dataset de prediccion de consumo eléctrico

    * **get_df** - Es la primera función que se usa para crear el tataset de entrenamiento. Lo que hace es coger cada uno de los dias dados en el json crudo y genera un **dataframe** donde tiene por cada fila un datetime con el valor de una fecha y una hora y en la segunda columna el valor de consumo. Es decir, para cada dia 24 filas en el df. y columnas "datetime" y "value".

    * **crear_ventanas** - *Funcion auxiliar utilizada dentro de la función principal process_data* para crear ventanas de dias contiguos, los del input y el de target, y el vector de condiciones para ese día del histórico. Finalmente se ha dejado tras la experimentación hadcodeado a un solo dia de datos previos. Sin embargo ya que ya ha sido programado, se deja la función por si en el futuro se quisiera ampliar con otros modelos.

    * **get_dataset** -   recibe como input 3 tuplas que contienen cada una el conjunto de datos ,**en numpy**, de train val y test.  Ej: train = (train_seq, train_cond, train_y).   Transforma cada tupla en un **tf.dataset (con dos inputs, y un target)**.

    * **process_data** - Es la funcion principal de procesado de los datos. 
        * **Inputs:** el **dataframe** llamado "df" contiene los datos de consumo con fechas redondeadas a horas. Es obtenido previamente por get_df, (en /views.py > # create_model), la **localizacion** (codigo del municipio donde se ubica el prosumidor), **use_predictions** (Bool para configurar si queremos que el modelo cruce los datos con el histórico de predicciones de AEMET que tenemos), **workday_df** (dataframe con los dias históricos dados indicando si fueron o no festivos).
        * Que hace:
            * Si "use_predictions" == True, carga el csv "localizacion".csv ubicado en /AEMET/predictions/ y lo transforma en un dataframe llamado "df_aemet"
            * Se quitan features (columnas) de df que tengan mas del 5% de NaNs. (Esto no se utiliza ahora mismo ya que de momento mi código solo utiliza el valor total de potencia usada por el prosumidor, no como en la experimentacion que usabamos mas de 10 features.) SIN EMBARGO, EN CASO DE QUERER EN EL FUTURO USAR MAS FEATURES, SIMPLEMENTE HABRÁ QUE EXTRAERLAS DEL JSON Y AÑADIRLAS EN LA FUNCION get_df. Para esta funcion es transparente y será capaz de utilizarlo adecuadamente. Si se le pasa un df con mas medidas internas de consumo de la empresa, las incluirá en el input de consumo del dia anterior, llamado en el código "X_seq".
            * Se filtran outliers por percentiles, los valores de consumo demasiado alejados se consideran errores. Este limite es arbitrario y sirvió para descartar errores en los datos de experimentacion
            * Se eliminan dias sin medidas para todas sus horas
            * Se escalan los consumos con Standard Scaler de scikit-learn (restamos media y dividimos entre la varianza si no recuerdo mal)
            * Se **guarda** las **deviacion típica** y la **media** de los consumos, para ser capaz de desescalar las predicciones del modelo y llevarlo a la escala real
            * Se añaden las features codificacion SIN COS de Dia de la Semana, codificacion SIN COS del Mes,  Variables dia anterior (consumo medio minimo y maximo).
            * Se hace un merge entre df y workday_df
            * Si use_predictions == True, Se hace un join entre el df de consumo, y el df_aemet con las predicciones. (Solo se conservan los dias del df que tambien esten presentes en el df de AEMET, por tanto si no hay predicciones para esas fechas de datos en ese municipio, el df estará vacío y fallará, en el caso de que el FLAG de use_weather/use_predictions sea TRUE)
            * Se rellenan los NaNs si los hubiera (aunque apenas debería haber)
            * Se calcula la variable **"variables_condicionales"**. Esta variable sirve para que a la hora de crear el dataset, la funcion "crear_ventanas" sepa que columnas pertenecen al "X_seq" (los datos de consumo del dia anterior), y cuales a "X_cond" (que son las características especiales que queremos que influyan en la prediccion, la prevision del clima del dia siguiente, si es o no dia laborable, y las demas features que he definido como el mes, dia de la semana...). De partida  variables_condicionales = 8   ya que mi codigo siempre añade las features (2 X SIN COS de Dia de la Semana, 2 X SIN COS de Mes, 3 X  Variables dia anterior, 1 X is_workday).
            Finalmente, si en este dataset se ha configurado usar las predicciones de aemet, se añade su número de features. *if use_predictions: variables_condicionales += df_aemet.shape[1]*
            * Se crean las ventanas de dias continuos que formaran nuestro dataset, separandolos en 3 numpys arrays "X_seq", "X_condition" y "y".  *X_seq, X_condition, y  = crear_ventanas(df, INPUT_DAYS, variables_condicionales)*.  Input days es la cantidad de dias previos de consumo que se le dan. Tras lo experimentos esta hardcodeado a "1", pero se puede cambiar en el futuro sin cabiar nada mas que la variable "INPUT_DAYS".
            * Divido el dataset en Train, Val y Test. Para test, ya que se trata de un problema de series temporales, he decidido no barajear el conjunto con el resto, por tanto en este punto separo el último 20% de los ejemplos de "X_seq" "X_condition" y "y" y los meto en una tupla () para llamarlo "test"
            * Barajeo el orden de los demas datos y separo en train y validacion. (Esto ha demostrado ser muy bueno para el modelo en los experimentos, ya que hay muchas diferencias de consumo por caracter estacional y si no el modelo veía muy pocas estaciones en train (la mitad), lo cual era un problema grave para generalizar...)
            * Se pasan estos conjuntos train, val y test a la fucnion *get_dataset* para que los transforme a tf.dataset y finalemente se devuelven los datasets.
            

    
    * **get_aemet_prediction** - Finalmente está "get_aemet_prediction", esta función es una excepción, ya que no se utiliza para crear el dataset de entrenamiento, sino para crear el vector input de inferencia, solo en el caso de que el modelo del prosumidor fuese entrenado con datos de predicciones históricas mediante el FLAG "use_weather". Es decir, solicita la prevision de mañana para un municipio (en el que se ubica el prosumidor, se saca de los atributos del modelo) a la AEMET para juntarlos con las otras features condicionales y usarlo junto los datos de consumo de hoy para la predicción.

* **¿¿ TO DO ?? views.py**
---
---

## Como usar la API
#### Despliegue local sin Docker
- acceder a ./src/django/
- Si no se tienen instaladas las dependecias de python3, se pueden instalar llamando a requirements.txt que es uno de los ficheros usados para crear el docker de django
```bash
python3 -m pip install -r requirements.txt
``` 
- Para levantar el servidor de django debemos ejecutar el siguiente comando. En mi caso todas las pruebas las he hecho con el puerto 1338, por ello las request de postman adjuntadas apuntan a ese puerto.
```bash
python3 manage.py runserver <puerto>
``` 

- A continuación podemos interactuar con la API a través de la interfaz de swagger o mediante el postman.
- Para acceder a swagger accedemos mediante HTTP a localhost con el puerto dado y '/docs/ mediante el navegador. En mi caso el 1338  ( http://localhost:1338/docs/ )

- Para obtener token, con swagger ejecutamos el endpoint para solicitar el token con las credenciales:    
Credenciales de testeo  -  user: inaki     password:123456
  la API responde un token en el campo 'access' de la respuesta.  Copiar ese token y pegarlo en la esquina superior derecha donde pone "Authorize"
  A partir de este momento, se podra hacer uso del resto de endpoint y automaticamente swagger adjunta el token de autenticacion en el header.

- Por ejemplo podemos probar a ver la lista de modelos de clasificacion disponibles en la API. para ello ejecutamos /clasifier/models_list/
En la respuesta vemos los modelos, entre los que está el modelo de clasificacion por defecto  'default_classifier'
- Podemos probar a ver su descripcion con  /classifier/model_description/  pasando como parámetro el nombre del 'default_classifier'
- tambien podemos probar a inferir la clasificacion con /classifier/inference/ o crear un clasificador dando como argumento los datos adecuados. 

EL formato y los datos requeridos se pueden consultar en el documento de interfaces (Aunque probablemente cambie en el futuro. Se añade el último a dia 6-11.2023). 

Tambien se puede probar, y va a ser mas cómodo, directamente con las request de postman adjuntadas. Todas ellas necesitarán un token nuevo de acceso. Lo bueno es que ya tienen un body
Adecuado y va a ser mas sencillo probar así los métodos. Se pueden importar al postman cargando el fichero "brain api.postman_collection"

Cabe destacar, que si se inicia con este método el servidor, se levanta en el localhost con HTTP.  Mis request estan todas puestas como HTTP. Si se levanta con docker,
La comunicacion se vuelve cifrada HTTPS tambien en el localhost. Lo único que habría que hacer es añadir a cada resquest del postman la "s" de https en el endpoint.

#### Despliege con Docker
- acceder a ./src/ 
- Iniciar daemon de docker
- levantar el docker:
```bash
docker compose up
``` 
- Por defecto he dejado el puerto 1338,  asi que acceder a " https://localhost:1338/docs/ " para visualizar la documentación de swagger.  Este puerto se puede cambiar en el docker compose, que se halla en /src/

Como se indica previamente, en caso de querer usar las request de postman, añadir "s" de https en cada request.  Igualmente será encesario solicitar un nuevo token con las credenciales.
Finalemente recordar que este puerto será accesible dentro de nuestra red cambiando localhost por nuestra IP dentro de la red local.
