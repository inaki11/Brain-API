import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_heuristicas(datos_horarios):

    perfil_de_referencia = np.mean(datos_horarios, axis=0)

    # Consumo medio diario máximo
    tmp = perfil_de_referencia.reshape(-1,24)
    consumo_medio_diario_maximo = np.max(np.mean(tmp, axis=1))
    print(f"consumo_medio_diario_maximo: {consumo_medio_diario_maximo:.2f}")

    # Consumo horario medio
    consumo_horario_medio = np.mean(perfil_de_referencia)
    print(f"consumo_horario_medio: {consumo_horario_medio:.2f}")

    # Consumo horario máximo
    consumo_horario_maximo = np.max(perfil_de_referencia)
    print(f"consumo_horario_maximo: {consumo_horario_maximo:.2f}")

    # Consumo horario mínimo
    consumo_horario_minimo = np.min(perfil_de_referencia)
    print(f"consumo_horario_minimo: {consumo_horario_minimo:.2f}")

    niveles_de_potencia = np.mean(tmp, axis=1) / consumo_medio_diario_maximo

    print("2")

    # Niveles de potencia lunes
    nivel_de_potencia_lunes = niveles_de_potencia[0] * 100
    print(f"nivel_de_potencia_lunes: {nivel_de_potencia_lunes:.2f}")

    # Niveles de potencia martes
    nivel_de_potencia_martes = niveles_de_potencia[1] * 100
    print(f"nivel_de_potencia_martes: {nivel_de_potencia_martes:.2f}")

    # Niveles de potencia miércoles
    nivel_de_potencia_miercoles = niveles_de_potencia[2] * 100
    print(f"nivel_de_potencia_miercoles: {nivel_de_potencia_miercoles:.2f}")

    # Niveles de potencia jueves
    nivel_de_potencia_jueves = niveles_de_potencia[3] * 100
    print(f"nivel_de_potencia_jueves: {nivel_de_potencia_jueves:.2f}")

    # Niveles de potencia viernes
    nivel_de_potencia_viernes = niveles_de_potencia[4] * 100
    print(f"nivel_de_potencia_viernes: {nivel_de_potencia_viernes:.2f}")

    print("3")

    # Niveles de potencia sabado
    nivel_de_potencia_sabado = niveles_de_potencia[5] * 100
    print(f"nivel_de_potencia_sabado: {nivel_de_potencia_sabado:.2f}")

    # Niveles de potencia domingo
    nivel_de_potencia_domingo = niveles_de_potencia[6] * 100
    print(f"nivel_de_potencia_domingo: {nivel_de_potencia_domingo:.2f}")

    media_por_dia = np.mean(tmp, axis=1)

    # Desviación de potencia lunes
    desciavion_potencia_lunes = np.mean(np.abs(tmp[0] - media_por_dia[0]) / np.mean(tmp[0])) * 100
    print(f"desciavion_potencia_lunes: {desciavion_potencia_lunes:.2f}")

    # Desviación de potencia martes * 100
    desciavion_potencia_martes = np.mean(np.abs(tmp[1] - media_por_dia[1]) / np.mean(tmp[1] )) * 100
    print(f"desciavion_potencia_martes: {desciavion_potencia_martes:.2f}")

    # Desviación de potencia miércoles
    desciavion_potencia_miércoles = np.mean(np.abs(tmp[2] - media_por_dia[2]) / np.mean(tmp[2])) * 100
    print(f"desciavion_potencia_miércoles: {desciavion_potencia_miércoles:.2f}")

    print("4")

    # Desviación de potencia jueves
    desciavion_potencia_jueves = np.mean(np.abs(tmp[3] - media_por_dia[3]) / np.mean(tmp[3])) * 100
    print(f"desciavion_potencia_jueves: {desciavion_potencia_jueves:.2f}")

    # Desviación de potencia viernes
    desciavion_potencia_viernes = np.mean(np.abs(tmp[4] - media_por_dia[4]) / np.mean(tmp[4])) * 100
    print(f"desciavion_potencia_viernes: {desciavion_potencia_viernes:.2f}")

    # Desviación de potencia sabado
    desciavion_potencia_sabado = np.mean(np.abs(tmp[5] - media_por_dia[5]) / np.mean(tmp[5])) * 100
    print(f"desciavion_potencia_sabado: {desciavion_potencia_sabado:.2f}")

    # Desviación de potencia domingo
    desciavion_potencia_domingo = np.mean(np.abs(tmp[6] - media_por_dia[6]) / np.mean(tmp[6])) * 100
    print(f"desciavion_potencia_domingo: {desciavion_potencia_domingo:.2f}")

    ######### media de la desviacion absoluta de cada dia de la semana respecto a su dia del perfil de referencia

    # Similitud de perfil de consumo los lunes
    datos_lunes = datos_horarios[:,0:24]
    perfil_de_referencia_lunes = perfil_de_referencia[0:24]
    numero_de_dias = datos_lunes.shape[0]

    similitud_perfil_consumo_lunes = (np.abs(datos_lunes - np.tile(perfil_de_referencia_lunes, numero_de_dias).reshape(-1,24)) /  np.tile(perfil_de_referencia_lunes, numero_de_dias).reshape(-1,24)  ).mean() * 100
    print(f"similitud_perfil_consumo_lunes:  {similitud_perfil_consumo_lunes:.2f}")


    # Similitud de perfil de consumo los martes
    datos_martes = datos_horarios[:,24:48]
    perfil_de_referencia_martes = perfil_de_referencia[24:48]

    similitud_perfil_consumo_martes = (np.abs(datos_martes - np.tile(perfil_de_referencia_martes, numero_de_dias).reshape(-1,24)) / np.tile(perfil_de_referencia_martes, numero_de_dias).reshape(-1,24) ).mean() * 100
    print(f"similitud_perfil_consumo_martes:  {similitud_perfil_consumo_martes:.2f}")


    # Similitud de perfil de consumo los miercoles
    datos_miercoles = datos_horarios[:,48:72]
    perfil_de_referencia_miercoles = perfil_de_referencia[48:72]

    similitud_perfil_consumo_miercoles = (np.abs(datos_miercoles - np.tile(perfil_de_referencia_miercoles, numero_de_dias).reshape(-1,24)) / np.tile(perfil_de_referencia_miercoles, numero_de_dias).reshape(-1,24) ).mean() * 100
    print(f"similitud_perfil_consumo_miercoles:  {similitud_perfil_consumo_miercoles:.2f}")


    # Similitud de perfil de consumo los jueves
    datos_jueves = datos_horarios[:,72:96]
    perfil_de_referencia_jueves = perfil_de_referencia[72:96]

    similitud_perfil_consumo_jueves = (np.abs(datos_jueves - np.tile(perfil_de_referencia_jueves, numero_de_dias).reshape(-1,24)) / np.tile(perfil_de_referencia_jueves, numero_de_dias).reshape(-1,24) ).mean() * 100
    print(f"similitud_perfil_consumo_jueves:  {similitud_perfil_consumo_jueves:.2f}")


    # Similitud de perfil de consumo los viernes
    datos_viernes = datos_horarios[:,96:120]
    perfil_de_referencia_viernes = perfil_de_referencia[96:120]

    similitud_perfil_consumo_viernes = (np.abs(datos_viernes - np.tile(perfil_de_referencia_viernes, numero_de_dias).reshape(-1,24)) / np.tile(perfil_de_referencia_viernes, numero_de_dias).reshape(-1,24) ).mean() * 100
    print(f"similitud_perfil_consumo_viernes:  {similitud_perfil_consumo_viernes:.2f}")


    # Similitud de perfil de consumo los sabado
    datos_sabado = datos_horarios[:,120:144]
    perfil_de_referencia_sabado = perfil_de_referencia[120:144]

    similitud_perfil_consumo_sabado = (np.abs(datos_sabado - np.tile(perfil_de_referencia_sabado, numero_de_dias).reshape(-1,24)) / np.tile(perfil_de_referencia_sabado, numero_de_dias).reshape(-1,24) ).mean() * 100
    print(f"similitud_perfil_consumo_sabado:  {similitud_perfil_consumo_sabado:.2f}")


    # Similitud de perfil de consumo los domingo
    datos_domingo = datos_horarios[:,144:168]
    perfil_de_referencia_domingo = perfil_de_referencia[144:168]

    similitud_perfil_consumo_domingo = (np.abs(datos_domingo - np.tile(perfil_de_referencia_domingo, numero_de_dias).reshape(-1,24)) / np.tile(perfil_de_referencia_domingo, numero_de_dias).reshape(-1,24) ).mean() * 100
    print(f"similitud_perfil_consumo_domingo:  {similitud_perfil_consumo_domingo:.2f}")



    # Puntos de Ruptura es una heurística diseñada por Aaron.


    # Primero sacamos los cambios de consumos entre horas, y el consumo maximo del perfil de referencia
    pendiente = np.append(- perfil_de_referencia[:-1] + perfil_de_referencia[1:], [0])
    max_potencia = np.max(perfil_de_referencia)
    max_pendiente = np.max(np.abs(pendiente))
    #print(pendiente)
        
    # A continuacion se saca Cambios_1, que busca aquellas posiciones donde cambia mas de un 6% el consumo    
    cambios_1 = (pendiente/max_pendiente > 0.06).astype(int) - (pendiente/max_pendiente < -0.06).astype(int)
    #print(cambios_1)
    # Despues donde se cambia de tendencia
    cambios_1 = np.where(cambios_1[:-1] != cambios_1[1:])[0] + 1
    #print(f"Cambios 1: {cambios_1}")


    # Despues se buscan todas las posiciones de cambio
    cambios_2 = (pendiente > 0).astype(int) - (pendiente <= 0).astype(int)
    #print(cambios_2)
    cambios_2 = np.where( (cambios_2[:-1] == -1) & (cambios_2[1:] == 1))[0] + 1
    #print(f"Cambios 2: {cambios_2}")


    # Filtramos los cambios_1
    #   Encontramos elementos comunes entre ambos, y si superan el umbral de la mitad de la potencia maxima, 
    #   los eliminamos de cambios 1
    elementos_comunes = np.array([element for element in cambios_1 if element in cambios_2 ])
    elementos_comunes_consumo_alto = np.array([el for el in elementos_comunes if perfil_de_referencia[el]/max_potencia > 0.5])
    #print(f"elementos comunes con  consumo > 50%:{elementos_comunes}")
    cambios_1 = np.setdiff1d(cambios_1, elementos_comunes)
    #print(f"Cambios 1 tras eliminar comunes con cambios 2: {cambios_1}")


    # Filtramos cambios 2
    #   Para ello eliminamos todos los elementos cuya potencia este por debajo del 50%
    cambios_2 = np.array([el for el in cambios_2 if perfil_de_referencia[el]/max_potencia > 0.5])
    #print(f"Cambios 2 tras filtrar por potencia: {cambios_2}")

    puntos_de_ruptura = np.append(cambios_1, cambios_2).astype(int)
    print(f"Puntos de ruptura: {puntos_de_ruptura}")



    # Separamos os puntos de ruptura de los demas puntos y los pintamos

    perfil = np.concatenate((perfil_de_referencia, np.arange(1,169))).reshape(2,-1)

    ptos_ruptura = perfil[:, puntos_de_ruptura]
    otros_ptos = perfil[:, -puntos_de_ruptura]

    otros_ptos = np.ones(168, dtype=bool)
    otros_ptos[puntos_de_ruptura] = False

    # Utilizar la máscara booleana para extraer los valores
    otros_ptos = perfil[:, otros_ptos]



    es_alto_consumo = ptos_ruptura.astype(int)[0] > max_potencia/2
    aux = np.append(ptos_ruptura.astype(int), es_alto_consumo)
    aux = aux.reshape(3,-1)


    def consumo_medio_trabajo_standby(perfil, aux):
        trabajo = np.array([])
        parado = np.array([])
        cambio = np.where(aux[2,:-1] != aux[2,1:])[0] + 1
        print(cambio)
        for i in range(1, len(cambio)):
            #print("cambio " + str(cambio[i] - 1))
            #print(aux[2, cambio[i] - 1])
            #print("desde " + str(aux[1,cambio[i - 1]]) + " hasta " + str(aux[1,cambio[i] - 1] + 1))
            if(aux[2, cambio[i] - 1]):
                trabajo = np.append(trabajo, perfil[ aux[1,cambio[i - 1]] : aux[1,cambio[i] - 1] + 1 ] )
            else:
                parado = np.append(parado, perfil[ aux[1,cambio[i - 1]] : aux[1,cambio[i] - 1] + 1 ])#
        #print(trabajo)
            
        return trabajo, parado
            


    # Calculamos la aparicion de puntos de ruptura en las horas del dia
    semana_completa = np.zeros((24)).astype(int)
    entre_semana = np.zeros((24)).astype(int)
    fin_de_semana = np.zeros((24)).astype(int)

    for p in puntos_de_ruptura:
        if p < 120:
            entre_semana[p % 24] +=1
        else:
            fin_de_semana[p % 24] += 1
            
        semana_completa[p % 24] += 1

    print("perfil")
    print(perfil_de_referencia)
    print("\naux")
    print(aux)

    consumos_trabajo, consumos_standby = consumo_medio_trabajo_standby(perfil_de_referencia, aux)

    # Consumo medio de trabajo
    consumo_medio_trabajo =  consumos_trabajo.mean()
    print(f"consumo medio trabajo: {consumo_medio_trabajo:.2f} KW")

    # Consumo medio de standby
    consumo_medio_standby = consumos_standby.mean()
    print(f"consumo medio standby: {consumo_medio_standby:.2f} KW")

    #########

    # Repetitividad Total
    valor_punto_semana_completa = 100/semana_completa.sum()
    valores = [0,0,0,0.40,0.60,0.80,1,1]
    repetitividad_semanal = np.array([el * valores[el] for el in semana_completa]).sum() * valor_punto_semana_completa
    print(f"rewpetitividad_semana_completa: {repetitividad_semanal:.2f}%")

    # Repetitividad laboral
    valor_punto_laboral = 100/entre_semana.sum()
    valores = [0, 0, 0, 0.5, 1, 1]   #  100% para 4 repeticiones o mas, 50% para 3 repeticiones 0% para menos.
    repetitividad_laboral = np.array([el * valores[el] for el in entre_semana]).sum() * valor_punto_laboral
    print(f"repetitividad_laboral: {repetitividad_laboral:.2f}%")


    # Repetitividad Fin de Semana
    valor_punto_fin_semana = 100/fin_de_semana.sum()
    valores = [0, 0, 1]   #  100% para 2 repeticiones, 0% para menos.
    repetitividad_fin_semana = np.array([el * valores[el] for el in fin_de_semana]).sum() * valor_punto_fin_semana
    print(f"repetitividad_fin_semana: {repetitividad_fin_semana:.2f}%")


    # Puntos de ruptura comunes
    puntos_ruptura_comunes = np.where(semana_completa>=4)
    print(f"Puntos de ruptura comunes: {puntos_ruptura_comunes}")


    # Numero de puntos laborales
    numero_puntos_laborables = entre_semana.sum()
    print(f"numero de puntos laborables: {numero_puntos_laborables}")

    # Número de puntos no laborales
    numero_puntos_fin_semana = fin_de_semana.sum()
    print(f"numero de puntos no laborables: {numero_puntos_fin_semana}")

    return  [consumo_medio_diario_maximo, consumo_horario_medio, consumo_horario_maximo, consumo_horario_minimo, nivel_de_potencia_lunes,
              nivel_de_potencia_martes, nivel_de_potencia_miercoles, nivel_de_potencia_jueves, nivel_de_potencia_viernes,
                nivel_de_potencia_sabado, nivel_de_potencia_domingo, desciavion_potencia_lunes, desciavion_potencia_martes,
                  desciavion_potencia_miércoles, desciavion_potencia_jueves, desciavion_potencia_viernes, desciavion_potencia_sabado,
                    desciavion_potencia_domingo, similitud_perfil_consumo_lunes, similitud_perfil_consumo_martes, similitud_perfil_consumo_miercoles,
                      similitud_perfil_consumo_jueves, similitud_perfil_consumo_viernes, similitud_perfil_consumo_sabado, similitud_perfil_consumo_domingo,
                        puntos_de_ruptura, consumo_medio_trabajo, consumo_medio_standby, repetitividad_semanal, repetitividad_laboral, repetitividad_fin_semana,
                          puntos_ruptura_comunes, numero_puntos_laborables, numero_puntos_fin_semana]