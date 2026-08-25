# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 11:18:15 2025

@author: hecto
"""

'''
Entrenamiento de modelo PINN para pronostico de PM2.5 en Ags
'''

#definimos las librerias que vamos a utilizar
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
#import numpy as np
import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split

#definimos la ruta donde esta la base de datos y la cargamos
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

#separamos los parametros de entrada y salida del modelo
#OJO, hay que convertilos a tensores de torch
x = torch.tensor(df.iloc[:,:-1].values, dtype = torch.float32)
y = torch.tensor(df.iloc[:,-1].values, dtype = torch.float32)

#definimos una clase para escalar los datos
class maxmin_scaler():
    def __init__(self, minimo, maximo):
        self.mini = minimo
        self.maxi = maximo
        self.max_dat = None
        self.min_dat = None
    
    #esta función escala
    def escalar(self, x):
        self.max_dat = x.max(axis=0).values
        self.min_dat = x.min(axis=0).values
        x_scal = self.mini + ((x - self.min_dat)*(self.maxi - self.mini))/(self.max_dat - self.min_dat)
        return x_scal
    
    #esta función des escala
    def escal_inver(self, x_scal):
        x = self.min_dat + (x_scal - self.mini)*(self.max_dat - self.min_dat)/(self.maxi - self.mini)
        return x

#generamos el objeto de escalador    
escalador = maxmin_scaler(-1, 1)

#escalamos los datos
x_scal = escalador.escalar(x)

#1ro generamos la estructura del modelo de red neuronal
class modelo_red(nn.Module):
    def __init__(self, entradas, neuronas, salidas):
        super().__init__()
        self.capa1 = nn.Linear(entradas, neuronas) #capa de entrada
        #self.norm1 = nn.LayerNorm(neuronas)
        #self.norm1 = nn.BatchNorm1d(neuronas)
        #self.capa2 = nn.Linear(neuronas, neuronas) #capa oculta 1
        self.capa3 = nn.Linear(neuronas, salidas) #capa oculta 2
        self.actfun = nn.LeakyReLU()
        #self.actfun = nn.Tanh() #aqui definimos la función de activación
        
    def forward(self, x):
        out = self.capa1(x)
        #out = self.norm1(out)
        out = self.actfun(out)
        #out = self.capa2(out)
        #out = self.actfun(out)
        out = self.capa3(out)
        return out

#creamos el modelo, el 1er argunmento son las entradas, 2do num neuronas y el útlimo las salidas
red = modelo_red(x_scal.shape[1], 300, 1)

#entrenamiento
mse = nn.MSELoss()
optimizador = optim.Adam(red.parameters(), lr = 0.001)
num_epocas = 10
perdida = []
perdida_val = []

#creamos el dataset
dataset = TensorDataset(x_scal,y)

#definimos el tamaño de el conjunto de datos de validación y enrenamiento
num_dat_entre = int(0.8 * len(dataset))
num_dat_val = len(dataset) - num_dat_entre

#dividimos los datos
train_dataset, val_dataset = random_split(dataset, [num_dat_entre, num_dat_val])

#creamos el dataloader en función del tamaño de lote
lotes = 1  #elemento en el lote
dataloader = DataLoader(train_dataset, batch_size = lotes, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size = lotes, shuffle=True)

for epocas in range(num_epocas):
    
    #definimos una lista para evalua el promedio de loss del lotes
    loss_lote  = 0
    red.train()
    
    #actualizamos los gradientes por lotes
    for batch_x, batch_y in dataloader:
        
        #evaluamos el modelo
        y_pred = red(batch_x)
        
        #despúes la pérdida
        loss = mse(y_pred, batch_y)
        
        #limpiamos los gradientes
        optimizador.zero_grad()
        
        #retropropagamos el error
        loss.backward()
        
        #actualizamos los pesos
        optimizador.step()
        
        #guardamos la predida del lote
        loss_lote += loss.item()
        
    #imprimos el error
    perdida.append(loss_lote/len(dataloader))
    print(f'Epoca: {epocas}, Pérdida_entrenamiento: {perdida[-1]:.4f}')
    
    #validación
    red.eval()
    loss_lote_val = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:                
            #evaluamos el modelo
            y_pred = red(batch_x)
            
            #evaluamos la pérdida
            loss = mse(y_pred, batch_y)
            
            #sumamos los errores
            loss_lote_val += loss.item()
    
    #evaluamos el promedio de la pérdida de la validación
    perdida_val.append(loss_lote_val/len(val_loader))
    
    #imprimimos los resultados por época
    print(f'Época: {epocas}, Pérdida_validación: {perdida_val[-1]:.4f}\n' )
    
    # if epocas > 0:
    #     serie.remove()
    # serie, = plt.plot(red(val_dataset.dataset.tensors[0]).detach().numpy(), val_dataset.dataset.tensors[1], 'o', alpha =0.4)
    # plt.pause(0.001)
    
#graficamos la pérdida durante el entrenamiento
plt.close('all')
plt.figure()
plt.plot(perdida, label = 'Entrenamiento')
plt.plot(perdida_val, label = 'Validación')
plt.legend()

#graficamos la respuesta del modelo contra los datos reales
# plt.figure()
# plt.plot(red(val_dataset.dataset.tensors[0]).detach().numpy(), val_dataset.dataset.tensors[1], 'o', alpha =0.4)
# plt.ylabel('Real')
# plt.xlabel('Modelo')