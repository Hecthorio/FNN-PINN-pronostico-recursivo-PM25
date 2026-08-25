# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 10:06:51 2025

@author: hecto
"""


'''
Script para evaluar la matriz de emisones. La matriz de emisiones contempla
tres tipos de emisiones: fuentes fijas, moviles, y área. Las emisiones por
fuentes fijas y de área se manejan como constantes, mientras que las de fuentes
moviles varian con el tiempo y se evaluan a partir de un modelo pre entrenado de 
red neuronal (FNN)

NOTAS: En los df que se utilizan las columnas de latitud y longitud deben tener 
como titulos: "Latitud" y "Longitud" (1ra mayuscula), para que el algoritmo 
pueda detectarlas
'''

#definimos librerias
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

#definimos la clase 
class matriz_emisiones():
    #construimos el objeto 
    def __init__(self, nodos):
        #definimos la ruta y los limites del sistema
        self.ruta = 'C:/Users/hecto/OneDrive/Documentos/ITA/amidiq 2026/'
        self.inc = 1/111
        self.lat_lim = [21.79818125851826-self.inc, 21.978493676558536+self.inc]
        self.lon_lim = [-102.37665302280578-self.inc, -102.1857655833361+self.inc]
        
        #guardamos el numero de nodos
        self.nodos = [nodos, nodos]
        
        #generamos la matriz de emisiones
        self.matriz = np.zeros(self.nodos)
        
        #evaluamos la distancia entre los nodos
        self.dis_nodo = [(self.lat_lim[1] - self.lat_lim[0])/self.nodos[0], (self.lon_lim[1] - self.lon_lim[0])/self.nodos[0]]
        
        #agregamos las fuentes fijas a la matriz de ceros
        self.fuentes_fijas()
        
        #agregamos las fuentes de área a las fuentes fijas
        self.fuentes_area()
        
        #cargamos el modelo de red de emisiones por fuentes fijas y 
        #definimos los indices donde se dan estas emisiones
        self.fuentes_moviles()
        
    #######################################################################
    #                          FUNCIONES AUXILIARES                       #
    #######################################################################
    #filtrar los puntos de emisiones que estan fuera del área de estudio
    def filtro_area(self, df):    
        df = df[(df['Latitud'] > self.lat_lim[0]) & (df['Latitud'] < self.lat_lim[1])]
        df = df[(df['Longitud'] > self.lon_lim[0]) & (df['Longitud'] < self.lon_lim[1])]
        return df
    #regresión lineal
    def interp_lin(self,x,y,x_int):
        y_int = y[0] + (x_int - x[0])*(y[1]-y[0])/(x[1]-x[0])
        return round(y_int).astype(int)
    #evaluación de que indice le corresponde en la matriz a la emisión
    def indices(self, df):    
        indices_xy = np.array([self.interp_lin(self.lon_lim,[0,self.nodos[0]-1],df.Longitud),
                               self.interp_lin(self.lat_lim,[0,self.nodos[0]-1],df.Latitud)]).T
        return indices_xy
    #definimos una función para evaluar el seno y coseno de la hora [0-23] y dia de la semana [0-6]
    def tiempo(self, hora, dia):
        x = np.zeros((1,4))
        x[0,0] = np.sin(hora*2*np.pi/24)
        x[0,1] = np.cos(hora*2*np.pi/24)
        x[0,2] = np.sin((dia + hora/24)*2*np.pi/7)
        x[0,3] = np.cos((dia + hora/24)*2*np.pi/7)
        return x
    
    ###########################################################################
    #               FUNCIÓN PARA EVALUACIÓN DE FUENTES FIJAS                  #
    ###########################################################################
    
    def fuentes_fijas(self):
        #cargamos el df con las fuentes fijas
        df = pd.read_csv(self.ruta + 'fuentes_fijas_bueno.csv')
        #filtramos los datos
        df = self.filtro_area(df)
        #evaluamos los indices
        indices_xy = self.indices(df)
        #con los indices acomodamos los valores de las emisiones en cada posición
        #de la matriz (recordar de sumar en la posicón si ya habia un valor anterior)
        for j,i in enumerate(indices_xy):
            self.matriz[i[1],i[0]] +=  df['emision'].iloc[j]
        #convertión de Ton/año a ug/h y por último dividimos entre el área del sistema
        #para obtener ug/h.m2
        self.matriz *= 1000*1000*1e6/(365*24)/(self.dis_nodo[0]*111/1*self.dis_nodo[1]*111/1)*(1/1000**2)
    
    ###########################################################################
    #               FUNCIÓN PARA EVALUAR LAS FUENTES DE ÁREA                  #
    ###########################################################################
    
    def fuentes_area(self):
        #definimos la ruta donde esta la posición de cada ladrillera
        df = pd.read_csv(self.ruta + 'INEGI_DENUE_26022025.csv', encoding='latin')
        #filtramos el df para usar solo las ladrilleras que esten dentro del área de estudio
        df = self.filtro_area(df)
        #definimos el núm de ladrilleras
        ladrilleras_n = 350
        ladrilleras_emis = 1080.85     #Ton/año, de PROAIRE

        #evaluamos las emisiones por cada ladrillera y hacemos conversión de Ton/año 
        #a ug/h, después dividimos entre el área superficiel obteniendo ug/h.m2
        ladrilleras_emis_u = ladrilleras_emis/ladrilleras_n*1000*1000*1e6/(365*24)/(self.dis_nodo[0]*111/1*self.dis_nodo[1]*111/1)*(1/1000**2)

        #evaluamos los indices donde cae la ubicación de cada ladrillera
        indices_xy = self.indices(df)

        #actualizamos la matriz de emisiones con las fuentes de área
        for i in indices_xy:
            self.matriz[i[1],i[0]] += ladrilleras_emis_u
            
    ###########################################################################
    #            FUNCIÓN PARA EVALUAR LOS INDICES DE FUENTES MOVILES          #
    ###########################################################################
    
    def fuentes_moviles(self):
        #1ro cargamos el modelo de red neuronal para estimación de fuentes moviles
        self.red = load_model(self.ruta + 'modelo_emisiones.h5')
        
        #cargamos las coordenadas de los puntos inmportantes de trajfico en la ciudad
        df = pd.read_csv(self.ruta + 'fuentes_moviles_coordenadas.csv')
        
        #evaluamos los indices donde van a caer las emisiones
        self.indices_xy_movil = self.indices(df)
    
    ###########################################################################
    #        FUNCIÓN PARA EVALUAR LA MATRIZ DE EMISIONES TEMPORAL             #
    ###########################################################################
    
    def evaluacion(self, hora, dia):
        #hacemos una copia del la matriz de emisiones para poder actualizarla en cada
        #nuevo periodo de tiempo con las emisiones de fuentes moviles
        matriz_ft = self.matriz.copy()

        #Evaluamos las emisiones por fuentes moviles que genera el modelo de red, despúes 
        #hacemos la conversión de Ton/h a ug/h, luego dividimos entre el área de nuestro 
        #sistema (conversión de °2 a km2 y depués a m2) y por último dividimos entre el núm de puntos de trafico
        moviles_emis_u = self.red(self.tiempo(hora,dia)).numpy()[0][0]*1000*1000*1e6/(self.dis_nodo[0]*111/1*self.dis_nodo[1]*111/1)/len(self.indices_xy_movil)*(1/1000**2)

        #agregamos a la matriz de emisiones las emisiones por fuentes moviles
        for i in self.indices_xy_movil:
            matriz_ft[i[1],i[0]] += moviles_emis_u
        return matriz_ft
       
#emisiones = matriz_emisiones(24)
#matriz_em = emisiones.evaluacion(2.5, 3.5)

#%%

# #definimos los límites del sistema (el factor es de 1° = 111km)
# inc = 1/111

# #definimos la ruta del archivo
# ruta = 'C:/Users/hecto/OneDrive/Documentos/ITA/amidiq 2026/'

# #definimos los limites de nuestro sistema
# lat_lim = [21.79818125851826-inc, 21.978493676558536+inc]
# lon_lim = [-102.37665302280578-inc, -102.1857655833361+inc]

# #definimos los nodos (tamaño de la matriz)
# nodos = [24,24]

# #generamos la matriz
# matriz = np.zeros(nodos)

# #evaluamos la distancia entre nodo y nodo
# #el 1er temino es lat uy el 2do lon
# dis_nodo = [(lat_lim[1] - lat_lim[0])/nodos[0], (lon_lim[1] - lon_lim[0])/nodos[0]]

# ###############################################################################
# #                               FUENTES FIJAS                                 #
# ###############################################################################
# #leemos el df con los valores de la fuentes fijas
# df = pd.read_csv(ruta + 'fuentes_fijas_bueno.csv')

# #1ro filtramos todos los datos que estan por arriba o por abajo de los limites
# #de la frontera del sistema
# def filtro_area(df,lat_lim,lon_lim):    
#     df = df[(df['Latitud'] > lat_lim[0]) & (df['Latitud'] < lat_lim[1])]
#     df = df[(df['Longitud'] > lon_lim[0]) & (df['Longitud'] < lon_lim[1])]
#     return df

# #filtramos usando la función de filtro
# df = filtro_area(df,lat_lim,lon_lim)

# #definimos la función de interpolación lineal para determinar la
# #posición en la matriz donde cae la emisión
# def interp_lin(x,y,x_int):
#     y_int = y[0] + (x_int - x[0])*(y[1]-y[0])/(x[1]-x[0])
#     return round(y_int).astype(int)

# # definimos una función para evaluar los indices (xy) donde cae la ubicación
# #de cada fuente
# def indices(df,nodos):    
#     indices_xy = np.array([interp_lin(lon_lim,[0,nodos[0]-1],df.Longitud),
#                            interp_lin(lat_lim,[0,nodos[0]-1],df.Latitud)]).T
#     return indices_xy

# #evaluamos los indices
# indices_xy = indices(df,nodos)

# #con los indices acomodamos los valores de las emisiones en cada posición
# #de la matriz (recordar de sumar en la posicón si ya habia un valor anterior)
# for j,i in enumerate(indices_xy):
#     matriz[i[1],i[0]] +=  df['emision'].iloc[j]

# #convertión de Ton/año a ug/h y por último dividimos entre el área del sistema
# #para obtener ug/h.m2
# matriz *= 1000*1000*1e6/(365*24)/(dis_nodo[0]*111/1*dis_nodo[1]*111/1)*(1/1000**2)

# ###############################################################################
# #                             FUENTES ÁREA                                    #
# ###############################################################################

# #Ahora cargamos la base de datos de las coordenas de los puntos donde
# #se encuentran las fuentes de área (sobre escribo la var de df)
# df = pd.read_csv(ruta + 'INEGI_DENUE_26022025.csv', encoding='latin')

# #filtramos el df para usar solo las ladrilleras que esten dentro del área de estudio
# df = filtro_area(df, lat_lim, lon_lim)

# #definimos el núm de ladrilleras totales en todo el estado y las emisiones
# #totales en todo el estado reportadas (se asume que cada ladrillera emite
# #la misma cantidad)
# ladrilleras_n = 350
# ladrilleras_emis = 1080.85     #Ton/año, de PROAIRE

# #evaluamos las emisiones por cada ladrillera y hacemos conversión de Ton/año 
# #a ug/h, después dividimos entre el área superficiel obteniendo ug/h.m2
# ladrilleras_emis_u = ladrilleras_emis/ladrilleras_n*1000*1000*1e6/(365*24)/(dis_nodo[0]*111/1*dis_nodo[1]*111/1)*(1/1000**2)

# #evaluamos los indices donde cae la ubicación de cada ladrillera
# indices_xy = indices(df,nodos)

# #actualizamos la matriz de emisiones con las fuentes de área
# for i in indices_xy:
#     matriz[i[1],i[0]] += ladrilleras_emis_u

# ###############################################################################
# #                           FUENTES MOVILES                                   #
# ###############################################################################
# #evaluación de las emisiones por fuentes moviles
# #1ro cargamos el modelo de red neuronal
# red = load_model(ruta + 'modelo_emisiones.h5')

# #cargamos las coordenadas de los puntos inmportantes de trajfico en la ciudad
# df = pd.read_csv(ruta + 'fuentes_moviles_coordenadas.csv')

# #definimos una función para evaluar el seno y coseno de la hora [0-23] y dia de la semana [0-6]
# def tiempo(hora, dia):
#     x = np.zeros((1,4))
#     x[0,0] = np.sin(hora*2*np.pi/24)
#     x[0,1] = np.cos(hora*2*np.pi/24)
#     x[0,2] = np.sin((dia + hora/24)*2*np.pi/7)
#     x[0,3] = np.cos((dia + hora/24)*2*np.pi/7)
#     return x

# #evaluamos los indices donde van a caer las emisiones
# indices_xy = indices(df,nodos)

# #hacemos una copia del la matriz de emisiones para poder actualizarla en cada
# #nuevo periodo de tiempo con las emisiones de fuentes moviles
# matriz_ft = matriz.copy()

# #Evaluamos las emisiones por fuentes moviles que genera el modelo de red, despúes 
# #hacemos la conversión de Ton/h a ug/h, luego dividimos entre el área de nuestro 
# #sistema y por último dividimos entre el núm de puntos de trafico
# moviles_emis_u = red(tiempo(0,0)).numpy()[0][0]*1000*1000*1e6/(dis_nodo[0]*111/1*dis_nodo[1]*111/1)/len(df)*(1/1000**2)

# #agregamos a la matriz de emisiones las emisiones por fuentes moviles
# for i in indices_xy:
#     matriz_ft[i[1],i[0]] += moviles_emis_u
