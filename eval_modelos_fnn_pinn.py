# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 13:21:09 2025

@author: hecto
"""

'''
Script para evaluar los modelos de FNN y PINN y comparrarlos con los datos
obtendios por el modelo de diferencias finitas
'''

#definimos las librerias que vamos a utilizar
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matriz_emisiones as me
import modulo_alt_mezcla as mam
import modulo_velocidad_viento as mvv
import joblib
import matplotlib.pyplot as plt
import contextily as ctx

#definimos el numero de nodos
nodos = [24,24]

#la relación "promedio" es 1°=111km
inc = 1/111

#definimos los limites de nuestro sistema
lat_lim = [21.79818125851826-inc, 21.978493676558536+inc]
lon_lim = [-102.37665302280578-inc, -102.1857655833361+inc]

#definimos el día del mes que se va a analizar 1-31
dia_mes = 10 #15,2

#definimos de que estacion vamos a tomar los datos
estacion = 'CBT'

#1ro vamos a definir la ruta de los multiples archivos de donde esta la velocidad
#del viento y la concentración de PM2.5 en la estaciones de monitoreo
ruta = 'C:/Users/hecto/OneDrive/Documentos/ITA/Posdoc Proyecto/articulo 3/DatosPM25/'

#aquí definimos en el diccionario la base datos de PM2.5 de la estación y la base de
#datos de RUOA que le corresponde de ese periodo de tiempo
base_datos = {'CBT':['Datos SINAICA - CBTIS - PM2.5 - 2022-06-01 - 2022-07-01.csv', '2022-06-agsc_minuto_L1.csv'],
              'CEN':['Datos SINAICA - Centro - PM2.5 - 2022-06-01 - 2022-07-01.csv', '2022-06-agsc_minuto_L1.csv'],
              'IED':['Datos SINAICA - Instituto Educativo - PM2.5 - 2022-12-01 - 2023-01-01.csv', '2022-12-agsc_minuto_L1.csv'],
              'SMA':['Datos SINAICA - Secretaría de Medio Ambiente - PM2.5 - 2023-02-01 - 2023-03-01.csv', '2023-02-agsc_minuto_L1.csv']}

#cargamos el df con la info de de la velocidad y dirección del viento
df = pd.read_csv(ruta + base_datos[estacion][1], encoding='latin1', skiprows=7)


#generamos un dataframe independiente para generar la base de datos que se 
#utiliza para evaluar la altura de la mezcla
df_alt = df*1

#ahora eliminamos las columnas que no nos sirvan para el analsis (T, humedad, etc)
df.drop(columns = ['°C', '%', 'm/s.1', 'deg.1', 'mm', 'hPa', 'W/m^2'], inplace = True)

#eliminamos las variables que no nos sirvan para la evaluación de h
df_alt.drop(columns = ['%', 'm/s.1', 'deg','deg.1', 'mm', 'hPa'], inplace = True)

#convertimos a variable tipo fecha la 1ra columna
#df['yyyy-mm-dd HH:MM:SS'] = pd.to_datetime(df['yyyy-mm-dd HH:MM:SS'], dayfirst = True)
#df_alt['yyyy-mm-dd HH:MM:SS'] = pd.to_datetime(df_alt['yyyy-mm-dd HH:MM:SS'], dayfirst = True)
df['yyyy-mm-dd HH:MM:SS'] = pd.to_datetime(df['yyyy-mm-dd HH:MM:SS'], format='mixed', dayfirst = True)
df_alt['yyyy-mm-dd HH:MM:SS'] = pd.to_datetime(df_alt['yyyy-mm-dd HH:MM:SS'], format='mixed', dayfirst = True)

#filtramso el día seleccionado
df = df[df['yyyy-mm-dd HH:MM:SS'].dt.day == dia_mes]
df_alt = df_alt[df_alt['yyyy-mm-dd HH:MM:SS'].dt.day == dia_mes]

#evaluamos que día de la semana es ese día y que día del año
dia = df['yyyy-mm-dd HH:MM:SS'].dt.day_of_week.iloc[0]
doy = df['yyyy-mm-dd HH:MM:SS'].dt.dayofyear.iloc[0]

#guardamos la fecha
fecha = df['yyyy-mm-dd HH:MM:SS'].iloc[0]
fecha = fecha.strftime('%Y-%m-%d')

#eliminamos la columna de fecha
df.drop(columns = ['yyyy-mm-dd HH:MM:SS'], inplace = True)

#convertimos la velocidad del viento en m/h
vo = df['m/s'].to_numpy()*3600
do = df['deg'].to_numpy()

#convertimos el df en un arreglo de numpy
df_alt = np.array(df_alt)

#cargamos en memoria es escalador y el modelo para velocidad de viento
ruta_mod_vel = 'C:/Users/hecto/OneDrive/Documentos/ITA/Posdoc Proyecto/articulo 3/'
red_viento = load_model(ruta_mod_vel + 'modelo_viento.h5')
scaler = joblib.load(ruta_mod_vel + 'scaler_viento.pkl')

#evaluamos la altura de la mezcla a partir de la función
h_mezcla = mam.altura_mezcla(df_alt)

#la pasamos a arreglo de numpy
h_mezcla = np.array(h_mezcla)

#los valores menores a cero los hacemos iguales a 150
h_mezcla[h_mezcla < 150] = 150

#definimos los tiempos como
dt = 1/60    #incremento tiempo (h)
tf = 14
n = 1
dh = 500    #incrementos en metros

#generamos el objeto para poder hacer la evaluación de las emisiones
emisiones = me.matriz_emisiones(nodos[0])

#cargamos los modelos de redes neuronales
PINN = load_model('PINN_model.h5')

#generamos los vectores para "x" y "y"
x = np.linspace(0, dh*nodos[0], num = nodos[0])
y = np.linspace(0, dh*nodos[1], num = nodos[1])

#generamos la malla de los valores
X,Y = np.meshgrid(x,y)

#convertimos las matrices X y Y a las respectivas coordenadas en latitud y longitud
Xi = (X-np.min(X))/(np.max(X) - np.min(X))*(np.max(lon_lim) - np.min(lon_lim)) + np.min(lon_lim)
Yi = (Y-np.min(Y))/(np.max(Y) - np.min(Y))*(np.max(lat_lim) - np.min(lat_lim)) + np.min(lat_lim)

###############################################################################
#ESTOS BLOQUES DE CODIGO SON PARA DEFINIR LA CONDICIÓN INICIAL (CI)
#leemos el df con los datos
df_con_i = pd.read_csv(ruta + base_datos[estacion][0], encoding='latin1')
#df = pd.read_csv(ruta + base_datos[estacion][0])

#eliminamos la 1ra fila del df
df_con_i.drop(index = 0, inplace = True)

#convertimos a formato de fecha
df_con_i.Fecha = pd.to_datetime(df_con_i.Fecha)

#filtramos solmente los datos del día que nos interesa
df_con_i = df_con_i[df_con_i.Fecha.dt.day == dia_mes]

#generamos la matriz de concentraciones
Co = np.zeros(nodos)

#dejamos las fronteras del sistema como ceros
Co[1:-1,1:-1] = np.full((nodos[0]-2,nodos[0]-2),df_con_i['Concentraciones horarias'].iloc[0])

#generamos un vector que define los niveles del diagrama de countorno
#levels = np.arange(0,30000,1000)
levels = np.arange(0.1,100,5)


#%%
for t in np.arange(0,tf,dt):
    
    #definimos los valores de h
    h = h_mezcla[n]
    
    #evaluamos la matriz de emisiones en este tiempo
    q = emisiones.evaluacion(t,dia)
    
    #evaluamos la velocidad del viento
    ws = mvv.mod_velo_vient(lon_lim, lat_lim, nodos[0], vo[n]*np.cos(np.deg2rad(do[n]))/3600, vo[n]*np.sin(np.deg2rad(do[n]))/3600, doy, t, np, red_viento, scaler)
    vox, voy = ws[:,0]*3600, ws[:,1]*3600
    vox, voy = vox.reshape(nodos[0], nodos[1]), voy.reshape(nodos[0], nodos[1])
    
    #hacemos la misma evaluación pero con un incremento 
    delta_grados = 0.00001 #el incremento en la lat y lon
    #i_ws = mvv.mod_velo_vient([a + 0.01 for a in lon_lim], [b + 0.01 for b in lat_lim], nodos[0], vo[n]*np.cos(np.deg2rad(do[n]))/3600, vo[n]*np.sin(np.deg2rad(do[n]))/3600, doy, t, np, red_viento, scaler)
    ix_ws = mvv.mod_velo_vient(list(map(lambda x: x + delta_grados, lon_lim)), lat_lim, nodos[0], vo[n]*np.cos(np.deg2rad(do[n]))/3600, vo[n]*np.sin(np.deg2rad(do[n]))/3600, doy, t, np, red_viento, scaler)
    iy_ws = mvv.mod_velo_vient(lon_lim, list(map(lambda x: x + delta_grados, lat_lim)), nodos[0], vo[n]*np.cos(np.deg2rad(do[n]))/3600, vo[n]*np.sin(np.deg2rad(do[n]))/3600, doy, t, np, red_viento, scaler)
    i_vox, i_voy = ix_ws[:,0]*3600, iy_ws[:,1]*3600
    i_vox, i_voy = i_vox.reshape(nodos[0], nodos[1]), i_voy.reshape(nodos[0], nodos[1])
    
    #evaluamos las derivadas OJO no es el mismo delta de incremento en grados que en m, checar esto
    #aplicamos un factor de conversión, 1°=111km ó lo que es lo mismo 1°=111000m
    d_vox, d_voy = (i_vox - vox)/(delta_grados*(111/1*1000)), (i_voy - voy)/(delta_grados*(111/1*1000))
    
    #acomodamos los datos para evaluar el modelo
    concatenado = np.concatenate((X[1:-1,1:-1].reshape(-1,1),
                                  Y[1:-1,1:-1].reshape(-1,1),
                                  np.repeat(t,(len(X)-2)**2).reshape(-1,1),
                                  vox[1:-1,1:-1].reshape(-1,1),
                                  voy[1:-1,1:-1].reshape(-1,1),
                                  d_vox[1:-1,1:-1].reshape(-1,1),
                                  d_voy[1:-1,1:-1].reshape(-1,1),
                                  np.repeat(h,(len(X)-2)**2).reshape(-1,1),
                                  q[1:-1,1:-1].reshape(-1,1),
                                  Co[1:-1,1:-1].reshape(-1,1),
                                  Co[0:-2,1:-1].reshape(-1,1),
                                  Co[2:,1:-1].reshape(-1,1),
                                  Co[1:-1,2:].reshape(-1,1),
                                  Co[1:-1,0:-2].reshape(-1,1)), axis = 1)
    
    #evaluación del modelo de red
    Ct = PINN(concatenado).numpy()
    
    #actualizamos los valores
    Co[1:-1,1:-1] = Ct.reshape(nodos[0]-2,nodos[0]-2)*1
    
    #aumentamos el contador
    n += 1
    
    #graficamos
    if t == 0:
        fig, ax = plt.subplots()
        ax.set_xlim(lon_lim)
        ax.set_ylim(lat_lim)
        ctx.add_basemap(ax, zoom=13, crs="EPSG:4326",
                        attribution=False,
                        source=ctx.providers.OpenStreetMap.HOT)
        contorno = ax.contourf(Xi,Yi,Co, cmap = 'jet', vmax = 100, vmin = 0.1, extend = 'max', levels = levels, alpha = 0.2)
        ax.set_title(f'Tiempo: {t:.2f} horas   $\\theta$: {do[n]}   Vel: {vo[n]/3600:.2f} m/s')
        cbar = plt.colorbar(contorno, ax=ax)
        cbar.set_label('Concentración PM$_{2.5}$ ($\mu$g/m$^3$)')
        ax.set_xlabel('Longitud')
        ax.set_ylabel('Latitud')
        ax.tick_params(axis='x', labelrotation=45)
        plt.pause(0.01)
    else:
        contorno.remove()
        contorno = ax.contourf(Xi,Yi,Co, cmap = 'jet', vmax = 100, vmin = 0.1, extend = 'max', levels = levels, alpha = 0.2)
        plt.title(f'Tiempo: {t:.2f} horas   $\\theta$: {do[n]}   Vel: {vo[n]/3600:.2f} m/s')
        plt.pause(0.01)

#ajustamos la figura al contenido
plt.tight_layout()
