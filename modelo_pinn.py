# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 11:52:39 2025

@author: hecto
"""

'''
Script para entrenameinto de modelo de red espacial con datos del modelo
diferencial
'''

#definimos las librerias que usaremos
import  pandas as pd
import keras 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import r2_score

#definimos la ruta donde esta la base de datos
ruta = 'C:/Users/hecto/OneDrive/Documentos/ITA/amidiq 2026/'
df = pd.read_csv(ruta + 'base_datos.csv')

#orden de los datos en el df
'''
[0]: distancia en x(m); longitud
[1]: distancia en y(m); latitud
[2]: tiempo(h)
[3]: vox(m/h)
[4]: voy(m/h)
[5]: dvoxx(m/h.m)
[6]: dvoyy(m/h.m)
[7]: h(m)
[8]: q(ug/m2.h)
[9]: Co(ug/m3)
[10]: Coj+1(ug/m3)
[11]: Coj-1(ug/m3)
[12]: Coi+1(ug/m3)
[13]: Coi-1(ug/m3)
[14]: C(ug/m3)    
'''

#separamos la base de datos
x = np.array(df.iloc[:,0:-1])
y = np.array(df.iloc[:,-1], ndmin =2).T

#generamos el modelo
def gen_model(x_in, neuronas):
    entradas = keras.Input(shape = (x_in.shape[1],))
    x = keras.layers.Dense(neuronas, activation = 'relu')(entradas)
    #x = keras.layers.Dense(neuronas, activation = 'relu')(x)
    #x = keras.layers.Dense(neuronas, activation = 'relu')(x)
    salida = keras.layers.Dense(1, activation = 'linear')(x)
    modelo = keras.Model(inputs = entradas, outputs = salida)
    return modelo
    
#generamos el modelo
red = gen_model(x, 350)

#compilamos el modelo de red
red.compile(
    optimizer = keras.optimizers.Adam(learning_rate=1e-3),
    loss = keras.losses.MeanSquaredError()
    )

#entrenamos el  modelo
perdida = red.fit(x, y, batch_size = 40, epochs = 100, validation_split = 0.1, shuffle = True)

#graficamos
plt.close('all')
plt.figure()
plt.plot(perdida.history['loss'], label = 'Entrenamiento')
plt.plot(perdida.history['val_loss'], label = 'Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida (MSE)')
plt.annotate('$\\mathcal{L}_{\\text{train}} = $' + f'{perdida.history['loss'][-1]:.4f}\n'
             '$\\mathcal{L}_{\\text{valid}} = $' + f'{perdida.history['val_loss'][-1]:.4f}', 
             xy = (100, perdida.history['loss'][-1]),
             xytext = (50,7500),
             fontsize = 12,
             arrowprops = dict(facecolor = 'black', shrink = 0.05, fc = 'white'),
             bbox = dict(facecolor='white', alpha = 0.5))
plt.legend()
plt.tight_layout()
plt.savefig(ruta + 'entrenamiento_FNN.png', dpi=300)


#evaluamos el modelo
y_predict = red(x)

# evaluamos coeficiente de determinación
R2 = r2_score(y, y_predict)

plt.figure()
plt.text(10, 45, f'Coeficiente\ndeterminación\n$R^2 = $ {R2:.4f}', bbox =dict(facecolor = 'white', alpha = 0.5), horizontalalignment='center',
        verticalalignment='center')
plt.hexbin(y_predict, y, gridsize=50, cmap='turbo', bins='log')
plt.xlim(min(y_predict.numpy().min(), y.min()), max(y_predict.numpy().max(), y.max()))
plt.ylim(min(y_predict.numpy().min(), y.min()), max(y_predict.numpy().max(), y.max()))
plt.colorbar(label = 'Densidad datos')
plt.plot([min(y_predict.numpy().min(), y.min()), max(y_predict.numpy().max(), y.max())], 
         [min(y_predict.numpy().min(), y.min()), max(y_predict.numpy().max(), y.max())], 
         color = 'red', ls = '--', label = 'Referencia')
plt.xlabel('FNN')
plt.ylabel('Datos')
plt.legend()
plt.tight_layout()
plt.savefig(ruta + 'val_FNN', dpi=300)

# plt.figure()
# plt.plot(red.predict(x), y, 'o', alpha = 0.5)
# plt.plot([0, max(y)[0]], [0, max(y)[0]], color = 'red', ls = '--')
# plt.xlabel('Concentración $\mu$g/m$^3$ (FNN)')
# plt.ylabel('Concentración $\mu$g/m$^3$ (Datos)')

#guardamos el modelo
#red.save(ruta + 'FNN_model.h5')

#%%
    
#generamos un nuevo modelo
red_pinn = gen_model(x, 100)

#definimos nuestro optimizador
optimizador = tf.keras.optimizers.Adam(learning_rate = 0.001)

#definimos la función de la PINN. NOTA, las variables de las cuales vamos a sacar
#las derivadas las pongo como argumentos a parte de entrada, depués las concateno para evaluar el modelo
def PINN(red, x, y, t, xp):
    
    # Convertir todo a tensores
    x  = tf.convert_to_tensor(x, dtype=tf.float32)
    y  = tf.convert_to_tensor(y, dtype=tf.float32)
    t  = tf.convert_to_tensor(t, dtype=tf.float32)
    xp = tf.convert_to_tensor(xp, dtype=tf.float32)
    
    #definimos cuales son las variables que se van a seguir para evaluiar sus 2das derivadas
    with tf.GradientTape(persistent=True) as tape1:
        #en esta parte definimos la variabes con respecto a las que vamos a 
        #evaluar la derivadas que son x (lon), y (lat) y t (tiempo)
        tape1.watch([x, y])
        
        #definimos que variables se van a seguir para evaluar su 1da derivada
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, y, t])
            
            # Forzar entrada al modelo
            entradas = tf.concat([x, y, t, xp], axis=1)
            
            #evaluamos el modelo, pero antes concatenamos todas las variable que
            #necesita el modelo de red neuronal para funcionar
            c = red(entradas)
        
        # print("x:", x.dtype, x.shape)
        # print("y:", y.dtype, y.shape)
        # print("t:", t.dtype, t.shape)
        # print("xp:", xp.dtype, xp.shape)

        #evaluamos las 1ras derivadas
        dc_dx = tape2.gradient(c, x)
        dc_dy = tape2.gradient(c, y)
        dc_dt = tape2.gradient(c, t)
        
    #ahora evaluamos las 2das derivadas
    d2c_dx2 = tape1.gradient(dc_dx, x)
    d2c_dy2 = tape1.gradient(dc_dy, y)
    
    # print("dc_dx:", dc_dx)
    # print("dc_dy:", dc_dy)
    # print("dc_dt:", dc_dt)
    # print("d2c_dx2", d2c_dx2)
    # print('d2c_dy2', d2c_dy2)

    #eliminamos los tape
    del tape1, tape2    

    #evaluamos nuestra función de pérdida
    dif = 9.3e-10*(3600/1)  #difusividad promedio de las PM2.5 (m2/h)
    u = 0.0004/100*3600     #velocidad de sedimentación (m/h)
    loss_pde = dc_dt +  xp[:,0:1]*dc_dx + xp[:,1:2]*dc_dy + c*(xp[:,2:3] + xp[:,3:4]) - (dif*(d2c_dx2 + d2c_dy2) + (xp[:,5:6] - c*u)/xp[:,4:5])
    
    #utilizmaos la función del error cuadratico medio
    loss_pde = tf.reduce_mean(tf.square(loss_pde))
    
    # PENALIZACIÓN por valores negativos
    #loss_no_negativo = tf.reduce_mean(tf.square(tf.nn.relu(-c)))  # Penaliza solo valores negativos
    
    return loss_pde #+ loss_no_negativo

# SOLUCIÓN: Usar un solo GradientTape no persistente
# def PINN(red, x, y, t, xp):
#     x = tf.convert_to_tensor(x, dtype=tf.float32)
#     y = tf.convert_to_tensor(y, dtype=tf.float32)
#     t = tf.convert_to_tensor(t, dtype=tf.float32)
#     xp = tf.convert_to_tensor(xp, dtype=tf.float32)
    
#     with tf.GradientTape(persistent=True) as tape2:
#         tape2.watch([x, y, t])
#         entradas = tf.concat([x, y, t, xp], axis=1)
#         c = red(entradas)
    
#     # Primeras derivadas
#     dc_dx = tape2.gradient(c, x)
#     dc_dy = tape2.gradient(c, y)
#     dc_dt = tape2.gradient(c, t)
    
#     # Segundas derivadas - calcular directamente
#     with tf.GradientTape() as tape_x:
#         tape_x.watch(x)
#         with tf.GradientTape() as tape_inner:
#             tape_inner.watch(x)
#             entradas_temp = tf.concat([x, y, t, xp], axis=1)
#             c_temp = red(entradas_temp)
#         dc_dx_temp = tape_inner.gradient(c_temp, x)
#     d2c_dx2 = tape_x.gradient(dc_dx_temp, x)
    
#     # Similar para d2c_dy2
#     with tf.GradientTape() as tape_y:
#         tape_y.watch(y)
#         with tf.GradientTape() as tape_inner:
#             tape_inner.watch(y)
#             entradas_temp = tf.concat([x, y, t, xp], axis=1)
#             c_temp = red(entradas_temp)
#         dc_dy_temp = tape_inner.gradient(c_temp, y)
#     d2c_dy2 = tape_y.gradient(dc_dy_temp, y)
    
#     del tape2, tape_x, tape_y, tape_inner
    
#     # Resto del código igual...
#     dif = 9.3e-10*(3600/1)
#     u = 0.0004/100*3600
#     loss_pde = dc_dt + xp[:,0:1]*dc_dx + xp[:,1:2]*dc_dy + c*(xp[:,2:3] + xp[:,3:4]) - (dif*(d2c_dx2 + d2c_dy2) + (xp[:,5:6] - c*u)/xp[:,4:5])
    
#     return tf.reduce_mean(tf.square(loss_pde))

#entrenamos el modelo, definimos las listas donde se guardara el historial de la función de pérdida
perdida_pde, perdida_datos = [], []
perdida_pde_val, perdida_datos_val = [], []
epocas = 100
batch_size = 20

from sklearn.model_selection import train_test_split

# Porcentaje de validación (por ejemplo, 10%)
val_split = 0.1
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_split, random_state=42)

# Convertir a tensores
x_train_t = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train_t = tf.convert_to_tensor(y_train, dtype=tf.float32)
x_val_t = tf.convert_to_tensor(x_val, dtype=tf.float32)
y_val_t = tf.convert_to_tensor(y_val, dtype=tf.float32)

# Convertimos los datos a tensores
#x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
#y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)

# Número total de batches
n_batches = int(np.ceil(len(x_train) / batch_size))

#convertimos la salida del modelo en un tensor
#yt = tf.convert_to_tensor(y, dtype=tf.float32)

#VERSIÓN OPTIMIZADA del entrenamiento
@tf.function  # Compilación para mejor rendimiento
def train_step(red, x_batch, y_batch, optimizador):
    with tf.GradientTape() as tape:
        # Separar características para PINN
        x_coord = x_batch[:, 0:1]
        y_coord = x_batch[:, 1:2] 
        t_coord = x_batch[:, 2:3]
        xp_features = x_batch[:, 3:]
        
        loss_pde = PINN(red, x_coord, y_coord, t_coord, xp_features)
        loss_datos = tf.reduce_mean(tf.square(red(x_batch) - y_batch))
        loss_total = loss_pde*0.1 + loss_datos
    
    gradientes = tape.gradient(loss_total, red.trainable_variables)
    # Gradient clipping más conservador
    #gradientes = [tf.clip_by_norm(g, 1.0) for g in gradientes if g is not None]
    #gradientes = [tf.clip_by_norm(g, 1.0) for g in gradientes]
    optimizador.apply_gradients(zip(gradientes, red.trainable_variables))
    
    return loss_pde, loss_datos, loss_total

@tf.function
def val_step(red, x_batch, y_batch):
    # Separar características para PINN
    x_coord = x_batch[:, 0:1]
    y_coord = x_batch[:, 1:2]
    t_coord = x_batch[:, 2:3]
    xp_features = x_batch[:, 3:]
    
    loss_pde = PINN(red, x_coord, y_coord, t_coord, xp_features)
    loss_datos = tf.reduce_mean(tf.square(red(x_batch) - y_batch))
    loss_total = loss_pde*0.1 + loss_datos
    
    return loss_pde, loss_datos, loss_total

# Antes del bucle, calcula el número de batches de validación
n_batches_val = int(np.ceil(len(x_val) / batch_size))

for i in range(epocas):
    # --- Entrenamiento (igual que antes) ---
    perdida_total_epoca = 0
    perdida_pde_epoca = 0
    perdida_datos_epoca = 0
    
    indices = np.random.permutation(len(x_train))
    x_mezclado = x_train_t.numpy()[indices]
    y_mezclado = y_train_t.numpy()[indices]
    
    for batch in range(n_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, len(x_train))
        
        x_batch = x_mezclado[start_idx:end_idx]
        y_batch = y_mezclado[start_idx:end_idx]
        
        loss_pde, loss_datos, loss_total = train_step(red_pinn, x_batch, y_batch, optimizador)
        
        perdida_total_epoca += loss_total.numpy()
        perdida_pde_epoca += loss_pde.numpy()
        perdida_datos_epoca += loss_datos.numpy()
    
    # Promedios de entrenamiento
    perdida_total_promedio = perdida_total_epoca / n_batches
    perdida_pde_promedio = perdida_pde_epoca / n_batches
    perdida_datos_promedio = perdida_datos_epoca / n_batches
    
    # Guardar entrenamiento
    perdida_pde.append(perdida_pde_promedio)
    perdida_datos.append(perdida_datos_promedio)
    
    # --- Validación ---
    perdida_pde_val_epoca = 0
    perdida_datos_val_epoca = 0
    perdida_total_val_epoca = 0
    
    for batch in range(n_batches_val):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, len(x_val))
        
        x_batch_val = x_val_t[start_idx:end_idx]
        y_batch_val = y_val_t[start_idx:end_idx]
        
        loss_pde_val, loss_datos_val_batch, loss_total_val = val_step(red_pinn, x_batch_val, y_batch_val)
        
        perdida_pde_val_epoca += loss_pde_val.numpy()
        perdida_datos_val_epoca += loss_datos_val_batch.numpy()
        perdida_total_val_epoca += loss_total_val.numpy()
    
    # Promedios de validación
    perdida_pde_val_promedio = perdida_pde_val_epoca / n_batches_val
    perdida_datos_val_promedio = perdida_datos_val_epoca / n_batches_val
    perdida_total_val_promedio = perdida_total_val_epoca / n_batches_val
    
    # Guardar validación
    perdida_pde_val.append(perdida_pde_val_promedio)
    perdida_datos_val.append(perdida_datos_val_promedio)
    
    # Imprimir progreso (incluyendo validación)
    print(f"Época {i}, Train Total: {perdida_total_promedio:.4e}, "
          f"Train PDE: {perdida_pde_promedio:.4e}, Train Datos: {perdida_datos_promedio:.4e} | "
          f"Val Total: {perdida_total_val_promedio:.4e}, Val PDE: {perdida_pde_val_promedio:.4e}, Val Datos: {perdida_datos_val_promedio:.4e}")

# for i in range(epocas):
#     perdida_total_epoca = 0
#     perdida_pde_epoca = 0
#     perdida_datos_epoca = 0
   
#     # Mezclar los datos al inicio de cada época
#     indices = np.random.permutation(len(x))
#     x_mezclado = x_tensor.numpy()[indices]
#     y_mezclado = y_tensor.numpy()[indices]
    
#     for batch in range(n_batches):
#         # Seleccionar el batch actual
#         start_idx = batch * batch_size
#         end_idx = min((batch + 1) * batch_size, len(x))
        
#         x_batch = x_mezclado[start_idx:end_idx]
#         y_batch = y_mezclado[start_idx:end_idx]
    
#         # with tf.GradientTape() as tape:
#         #     #evaluamos la función de que define la PINN para sacar la pérdida y la guardamos en la lista
#         #     loss_pde = PINN(red_pinn, x_batch[:,0:1], x_batch[:,1:2], x_batch[:,2:3], x_batch[:,3:])
#         #     perdida_pde.append(loss_pde.numpy())
#         #     #evaluamos el modelo, lo comparamos con los datos reales y evaluamos nuestra pérdida
#         #     loss_datos = tf.reduce_mean(tf.square(red_pinn(x_batch) - y_batch))
#         #     perdida_datos.append(loss_datos.numpy())
#         #     #definimos una variable como la pérdida total
#         #     loss_total = loss_pde + loss_datos

#         # #evaluamos los gradientes
#         # gradientes = tape.gradient(loss_total, red_pinn.trainable_variables)
#         # gradientes = [tf.clip_by_norm(g, 1.0) for g in gradientes]  # Gradient clipping
#         # optimizador.apply_gradients(zip(gradientes, red_pinn.trainable_variables))
        
#         # # Acumular pérdidas para el reporte
#         # perdida_total_epoca += loss_total.numpy()
#         # perdida_pde_epoca += loss_pde.numpy()
#         # perdida_datos_epoca += loss_datos.numpy()
        
#         loss_pde, loss_datos, loss_total = train_step(red_pinn, x_batch, y_batch, optimizador)
        
#         # Acumular pérdidas para el reporte
#         perdida_total_epoca += loss_total.numpy()
#         perdida_pde_epoca += loss_pde.numpy()
#         perdida_datos_epoca += loss_datos.numpy()
        
#     # Calcular promedios de la época
#     perdida_total_promedio = perdida_total_epoca / n_batches
#     perdida_pde_promedio = perdida_pde_epoca / n_batches
#     perdida_datos_promedio = perdida_datos_epoca / n_batches
    
#     # Guardar historial
#     perdida_pde.append(perdida_pde_promedio)
#     perdida_datos.append(perdida_datos_promedio)
    
#     # Imprimir progreso
#     print(f"Época {i}, Pérdida Total: {perdida_total_promedio:.4e}, "
#           f"PDE: {perdida_pde_promedio:.4e}, Datos: {perdida_datos_promedio:.4e}")

    #imprimimos la pérdida durante el entrenamiento de ambas funciones
    #print(f"Época {i}, Pérdida Total: {loss_total.numpy()}, PDE: {loss_pde.numpy()}, Datos: {loss_datos.numpy()}")

#Entrenamiento optimizado
# def entrenar_pinn_optimizado(red, x, y, epocas=100, batch_size=30):
#     perdida_pde_hist, perdida_datos_hist = [], []
    
#     # Preprocesar datos
#     #x_normalized, y_normalized, stats = preprocesar_datos(x, y)
#     x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
#     y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
    
#     n_batches = int(np.ceil(len(x) / batch_size))
    
#     for epoca in range(epocas):
#         # Mezclar datos
#         indices = np.random.permutation(len(x))
#         x_mezclado = tf.gather(x_tensor, indices)
#         y_mezclado = tf.gather(y_tensor, indices)
        
#         perdida_pde_epoca, perdida_datos_epoca = 0, 0
        
#         for batch in range(n_batches):
#             start_idx = batch * batch_size
#             end_idx = min((batch + 1) * batch_size, len(x))
            
#             x_batch = x_mezclado[start_idx:end_idx]
#             y_batch = y_mezclado[start_idx:end_idx]
            
#             loss_pde, loss_datos, _ = train_step(red, x_batch, y_batch, optimizador)
            
#             perdida_pde_epoca += loss_pde.numpy()
#             perdida_datos_epoca += loss_datos.numpy()
        
#         # Guardar promedios
#         perdida_pde_hist.append(perdida_pde_epoca / n_batches)
#         perdida_datos_hist.append(perdida_datos_epoca / n_batches)
        
#         print(f"Época {epoca}, PDE: {perdida_pde_hist[-1]:.4e}, Datos: {perdida_datos_hist[-1]:.4e}")
    
#     return perdida_pde_hist, perdida_datos_hist

# perdida_pde, perdida_datos = entrenar_pinn_optimizado(red_pinn, x, y)

plt.figure()
plt.plot(perdida_pde, label='PDE Entrenamiento')
plt.plot(perdida_pde_val, label='PDE Validación')
plt.plot(perdida_datos, label='Datos Entrenamiento')
plt.plot(perdida_datos_val, label='Datos Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida (MSE)')
plt.annotate('$\\mathcal{L}_{\\text{train}}^{\\text{PDE}} = $' + f'{perdida_pde[-1]:.4f}\n'
             '$\\mathcal{L}_{\\text{valid}}^{\\text{PDE}} = $' + f'{perdida_pde_val[-1]:.4f}\n'
             '$\\mathcal{L}_{\\text{train}}^{\\text{Datos}} = $' + f'{perdida_datos[-1]:.4f}\n'
             '$\\mathcal{L}_{\\text{valid}}^{\\text{Datos}} = $' + f'{perdida_datos_val[-1]:.4f}', 
             xy = (100, perdida_pde[-1]),
             xytext = (30,40000),
             fontsize = 12,
             arrowprops = dict(facecolor = 'black', shrink = 0.05, fc = 'white'),
             bbox = dict(facecolor='white', alpha = 0.5))
plt.legend()
plt.tight_layout()
plt.savefig(ruta + 'entrenamiento_PINN.png', dpi=300)
#plt.title('Pérdida PDE')

# plt.figure()
# plt.plot(perdida_datos, label='Datos Entrenamiento')
# plt.plot(perdida_datos_val, label='Datos Validación')
# plt.xlabel('Épocas')
# plt.ylabel('Pérdida (MSE)')
# plt.legend()
# plt.title('Pérdida Datos')

# #graficamos las funciones de pérdida durante el entrenamiento
# #plt.close('all')
# plt.figure()
# plt.plot(perdida_pde, label = 'PDE')
# plt.xlabel('Épocas')
# plt.ylabel('Pérdida (MSE)')
# plt.legend()

# plt.figure()
# plt.plot(perdida_datos, label = 'Datos')
# plt.xlabel('Épocas')
# plt.ylabel('Pérdida (MSE)')
# plt.legend()

# plt.figure()
# plt.plot(red_pinn(x_val), y_val, 'o', alpha = 0.3)
# plt.plot([0, max(y)[0]], [0, max(y)[0]], color = 'red', ls = '--')
# plt.xlabel('Concentración $\mu$g/m$^3$ (PINN)')
# plt.ylabel('Concentración $\mu$g/m$^3$ (Datos)')
# plt.tight_layout()

#evaluamos el modelo
y_predict = red_pinn(x_train)

# evaluamos coeficiente de determinación
R2 = r2_score(y_train, y_predict)

plt.figure()
plt.text(10, 45, f'Coeficiente\ndeterminación\n$R^2 = $ {R2:.4f}', bbox =dict(facecolor = 'white', alpha = 0.5),
         horizontalalignment = 'center', verticalalignment = 'center')
plt.hexbin(y_predict, y_train, gridsize=50, cmap='turbo', bins='log')
plt.xlim(min(y_predict.numpy().min(), y_train.min()), max(y_predict.numpy().max(), y_train.max()))
plt.ylim(min(y_predict.numpy().min(), y_train.min()), max(y_predict.numpy().max(), y_train.max()))
plt.colorbar(label = 'Densidad datos')
plt.plot([min(y_predict.numpy().min(), y_train.min()), max(y_predict.numpy().max(), y_train.max())], 
         [min(y_predict.numpy().min(), y_train.min()), max(y_predict.numpy().max(), y_train.max())], 
         color = 'red', ls = '--', label = 'Referencia')
plt.xlabel('PINN')
plt.ylabel('Datos')
plt.legend()
plt.tight_layout()
plt.savefig(ruta + 'val_PINN.png', dpi=300)


#guardamos el modelo
#red_pinn.save(ruta + 'PINN_model.h5')

#a = PINN(red_pinn, x[:,0:1], x[:,1:2], x[:,2:3], x[:,3:])