# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:29:25 2025

@author: hecto
"""

#modulo para estimar la altura de la mezcla a aprtir de los parametros meteorologicos
def altura_mezcla(df):
    #importamos librerias
    import numpy as np
    
    #convertimos la temperatura de °C a K
    df[:,1] = df[:,1]+273.15
    
    #Convertimos los valores de rad de neg a ceros
    df[df[:,3] < 0, 3]=0
    
    #definimos algunos parametros importantes
    cp = 1005  #J/kg.K
    rho = 1.2  #kg/m3
    g = 9.81   #m/s2
    zo = 0.3   #longitud de rugosidad de la superficie para Ags
    k = 0.4    #constante de Von-Karman
    lat_ags = 21.88*np.pi/180
    z = 10     #m
    d = 5*zo

    #evaluamos el parametro de coriolis
    f = 2*7.2921e-5*np.sin(lat_ags)

    #definimos las funciones para condiciones
    def u_neutra(zo,k,u,z,d):
        #evaluación de la velocidad de arrastre
        u_ = k*u*1/(np.log(z/zo))
        return u_

    def u_inestable(zo,k,phi,u,z,L):
        #evaluamos la velociad de arrastre
        u_ = k*u*1/(np.log(z/zo) - phi*(z/L) + phi*(zo/L))
        return u_

    def Monin_Obukhov(rho,cp,T,u_,k,g,H):
        L = -rho*cp*T*(u_**3)/(k*g*H)
        return L

    def momentum_stability(z,L):
        x = (1-16*z/L)**(1/4)
        phi = 2*np.log((1+x)/2) + np.log((1+x**2)/2) - 2*np.arctan(x) + np.pi/2
        #phi = 2*np.log((1+x)/2) + np.log((1+x**2)/2) - 2/np.tan(x) + np.pi/2
        return phi

    def altura_mez(u_,f):
        h = 0.2*u_/f
        return h

    def altura_mez_noc(u):
        h = 125*u
        return h

    #definimos los parametros para estimar el flujo de calor sensible y los modelos
    r = 0.3             #albedo de la ciudad
    sigma = 5.67e-8     #cte de Stefan-Boltzmann (W/m^2.K^4)
    N = 0.5             #porcejate de nubocidad
    c1 = 5.31e-13       #cte empirica (W/m^2.K^6)
    c2 = 60             #cte empirica (W/m^2)
    c3 = 0.12           #cte empirica(adimensional)

    #radiación neta en la superficie
    def rad_neta(r,K,c1,T,sigma,c2,N,c3):
        Q = ((1-r)*K + c1*T**6 - sigma*T**4 + c2*N)/(1 + c3)
        return Q

    #modelo con la relación cp/calor latente de vaporización de agua con
    #respecto a la saturación especifica de humedad en el aire (dqs/dT)
    def gama_s(T):
        gamma = 1.4321*np.exp(-0.056*T)
        return gamma

    def calor_sen(T,Q):
        alpha = 0.45
        beta = 20 
        c_g = 0.1   
        G = c_g*Q   #flujo de calor hacia el suelo
        H = (1 - alpha + gama_s(T))/(1+gama_s(T))*(Q - G) - beta
        return H

    #definimos una variable donde se ira guardando los valores estimados de h
    h = []

    #definimos un ciclo para ir evaluando cada uno de los parametros
    for i in df:
        if i[3] < 20:
            u_ = u_neutra(zo, k, i[2], z, d)
            h.append(altura_mez_noc(i[2]))
        else:
            #proponemos valores iniciales
            error = 100
            #phi = 0
            Lo = -1000
            n = 0
            delta = 0.001
            #evaluamos el flujo de calor sensible
            Q = rad_neta(r, i[3], c1, i[1], sigma, c2, N, c3)
            H = calor_sen(i[1], Q)
            while error > 0.001:
                #evaluamos el flujo de calor sensible
                #Q = rad_neta(r, i[3], c1, i[1], sigma, c2, N, c3)
                #H = calor_sen(i[1], Q)
                for j in range(2):
                    #re evaluamos el valor de phi
                    phi = momentum_stability(z, Lo)
                    #evaluamos la velocidad de arrastre 
                    u_ = u_inestable(zo,k,phi,i[2],z,Lo)
                    #despues evaluamos la longitud de Momin
                    L = Monin_Obukhov(rho, cp, i[1], u_, k, g, H)
                    #evaluamos la función en cero para la función normal y la derivada
                    if j == 0:
                        ff = L - Lo
                        #aplicamos un incremento para evaluar en la 2da vuelta la derivada
                        Lo = Lo + delta
                    else:
                        #evaluamos la función con el incremento
                        f_inc = L - Lo
                        #evaluamos la derivada
                        fp = (f_inc - ff)/delta
                #aplicamos el NR
                L = Lo - delta - ff/fp*.01
                #evaluamos el error
                error = abs((L-Lo-delta)/L)
                #actualimos el valor de L
                Lo = L*1
                #actualizamos el contador
                n = n + 1
                print(L)
                if n > 200:
                    print('Se alcanzo el núm máx de iteraciones')
                    break
            #evaluamos el nuevo valor 
            h.append(altura_mez(u_, f))
    return h