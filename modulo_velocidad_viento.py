# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 13:04:19 2025

@author: hecto
"""

'''
Modulo para evaluar la velocidad de viento en toda la superficie del sistema usando
un modelo de FNN que necesita los limites del sistema
'''

def mod_velo_vient(x_lon, y_lat, nodos, vx_r, vy_r, doy, hour, np, red, scaler):
    #importamos librerias
    #import numpy as np
    #from tensorflow.keras.models import load_model
    
    #definimos los limites del sistema
    #x_lon = np.array([-102.37568491447393, -102.20035522245273])
    #y_lat = np.array([21.788600065285202, 21.97505731345356583])
    
    #definimos el numero de nodos
    #nodos = 10
    #horas = 100
    #indice = 10000
    
    #generamos el arreglo vacio que contendra toda la información para hacer las
    #evaluaciones del modelo con dimensiones en función del núm de nodos
    #OJO, las dimensiones son numero de datos y la columnas parametros o caracteristicad del modelo
    x = np.zeros([nodos*nodos, 10])
    
    #generamos los puntos donde se va a evaluar el modelo
    malla = np.meshgrid(np.linspace(x_lon[0], x_lon[1], nodos), np.linspace(y_lat[0], y_lat[1], nodos))
    #malla_lon, malla_lat = np.meshgrid(np.linspace(x_lon[0], x_lon[1], nodos), np.linspace(y_lat[0], y_lat[1], nodos))
    
    #agregamos los valores de lat y lon de los nodos donde se va a evaluar el modelo
    #x[:,4], x[:,5] = malla_lat.reshape(-1,1)[:,0], malla_lon.reshape(-1,1)[:,0]
    x[:,4], x[:,5] = malla[1].reshape(-1,1)[:,0], malla[0].reshape(-1,1)[:,0]
    
    #agregamos la lat y lon de la RUOA
    x[:,6], x[:,7] = np.repeat(21.91592292345118, nodos*nodos), np.repeat(-102.3190151576727, nodos*nodos)
    
    #agregamos los valores de la velocidad del viento (componentes) de la RUOA
    x[:,8], x[:,9] = np.repeat(vx_r, nodos*nodos), np.repeat(vy_r, nodos*nodos)
    
    #por último modificamos la variables temporales
    x[:,0], x[:,1] = np.repeat(np.sin(2*np.pi*hour/24), nodos*nodos), np.repeat(np.cos(2*np.pi*hour/24), nodos*nodos)
    x[:,2], x[:,3] = np.repeat(np.sin(2*np.pi*(doy + hour/24)/365), nodos*nodos), np.repeat(np.cos(2*np.pi*(doy + hour/24)/365), nodos*nodos)
    
    #reacomoddamos de mayor a menor las lat y lon porque streamplot lo requiere así
    #x = x[np.lexsort((x[:,4],x[:,5]))]
    
    #evaluamos el modelo
    v = red(scaler.transform(x)).numpy()
    
    return v