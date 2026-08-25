# FNN vs. PINN para la predicción recursiva espacio-temporal de PM₂.₅

Este repositorio contiene los códigos desarrollados para el **entrenamiento y evaluación de modelos de redes neuronales convencionales y redes neuronales informadas por física (Physics-Informed Neural Networks, PINN)** aplicadas a la estimación espacio-temporal de concentraciones de material particulado fino (**PM₂.₅**).

El objetivo principal es estudiar el efecto de incorporar conocimiento físico sobre los procesos de **transporte, dispersión, emisión y sedimentación atmosférica** dentro del entrenamiento de una red neuronal y analizar su comportamiento durante predicciones recursivas en el tiempo.

Se consideran dos enfoques:

* **FNN** — Feedforward Neural Network.
* **PINN** — Physics-Informed Neural Network.

Ambos modelos utilizan como referencia datos generados a partir de un modelo numérico de transporte y dispersión.

---

# Objetivo

Una red neuronal convencional aprende exclusivamente a partir de la relación existente entre las variables de entrada y los datos de salida.

Una PINN incorpora adicionalmente información física mediante una función de pérdida que penaliza las predicciones que no satisfacen aproximadamente la ecuación diferencial que describe el fenómeno.

El planteamiento general es:

```text
                    Datos del modelo físico
                            │
                    ┌───────┴────────┐
                    │                │
                    ▼                ▼
                   FNN              PINN
                    │                │
          Pérdida basada       Pérdida basada
             en datos          en datos + PDE
                    │                │
                    └───────┬────────┘
                            │
                            ▼
                 Predicción de PM2.5
                            │
                            ▼
                Evaluación recursiva
                            │
                            ▼
               Propagación temporal
                  de las predicciones
```

---

# Datos utilizados

Los modelos utilizan una base de datos denominada:

```text
base_datos.csv
```

La información contiene variables espaciales, temporales, meteorológicas y físicas asociadas con el transporte de PM₂.₅.

Las columnas utilizadas tienen el siguiente significado:

```text
x       : coordenada espacial x (m)
y       : coordenada espacial y (m)
t       : tiempo (h)

vox     : componente x de la velocidad del viento
voy     : componente y de la velocidad del viento

dvoxx   : derivada espacial de vox
dvoyy   : derivada espacial de voy

h       : altura de la capa de mezcla
q       : tasa de emisión

Co      : concentración en el nodo actual
Coj+1   : concentración en el nodo vecino
Coj-1   : concentración en el nodo vecino
Coi+1   : concentración en el nodo vecino
Coi-1   : concentración en el nodo vecino

C       : concentración objetivo
```

Por lo tanto, la red no solamente recibe información meteorológica y de emisiones, sino también información sobre el estado espacial de la concentración en el instante anterior.

---

# Interpretación autoregresiva del modelo

La formulación puede interpretarse como un modelo dinámico de un paso:

```text
C(t), vecinos(t), meteorología(t), emisiones(t)
                       │
                       ▼
                  Red neuronal
                       │
                       ▼
                  C(t + Δt)
```

Después de obtener la concentración para el siguiente instante, la predicción puede reutilizarse como nueva condición de entrada:

```text
Ĉ(t+Δt)
   │
   ▼
Entrada del siguiente paso
   │
   ▼
Ĉ(t+2Δt)
   │
   ▼
Ĉ(t+3Δt)
   │
   ▼
...
```

Esto constituye una **predicción recursiva o autoregresiva**.

No corresponde estrictamente a un modelo AR(1) estadístico clásico, debido a que la concentración futura depende no solamente de la concentración previa, sino también de variables meteorológicas, espaciales, emisiones y concentraciones de nodos vecinos.

Puede interpretarse de manera más general como:

```text
C(t+Δt) = f(
    C(t),
    C_vecinos(t),
    meteorología(t),
    emisiones(t),
    posición,
    tiempo
)
```

---

# Propagación del error

Durante una evaluación de un solo paso, el modelo utiliza como entrada valores conocidos.

Sin embargo, durante una predicción recursiva las salidas del modelo se convierten en entradas de los pasos siguientes.

Por lo tanto:

```text
Error en t+Δt
      │
      ▼
Entrada ligeramente incorrecta
en t+2Δt
      │
      ▼
Nuevo error
      │
      ▼
Acumulación / propagación
del error
```

Este comportamiento es especialmente importante en modelos utilizados para simulaciones temporales de largo horizonte.

Una de las motivaciones para utilizar una PINN es estudiar si la incorporación de restricciones físicas durante el entrenamiento ayuda a producir una dinámica más estable y físicamente consistente durante este tipo de propagación recursiva.

---

# Modelo FNN

La primera arquitectura corresponde a una **Feedforward Neural Network convencional**.

Su estructura es:

```text
Variables de entrada
        │
        ▼
Dense
350 neuronas
ReLU
        │
        ▼
Dense
1 neurona
Linear
        │
        ▼
Concentración PM2.5
```

El modelo se entrena utilizando:

```text
Optimizador     : Adam
Learning rate   : 0.001
Función pérdida : MSE
Batch size      : 40
Épocas          : 100
Validación      : 10 %
```

La función objetivo de la FNN es exclusivamente:

```text
L_FNN = L_datos
```

donde la pérdida de datos corresponde al error cuadrático medio entre las concentraciones predichas y los valores de referencia.

---

# Modelo PINN

La segunda arquitectura corresponde a una **Physics-Informed Neural Network**.

Su estructura base es similar a la FNN:

```text
Variables de entrada
        │
        ▼
Dense
100 neuronas
ReLU
        │
        ▼
Dense
1 neurona
Linear
        │
        ▼
Concentración PM2.5
```

La diferencia fundamental se encuentra en su proceso de entrenamiento.

Además del error con respecto a los datos, la PINN incorpora una segunda función de pérdida asociada con la ecuación diferencial que describe el transporte de PM₂.₅.

---

# Ecuación física

La restricción física considera procesos relacionados con:

* Variación temporal de la concentración.
* Transporte advectivo.
* Transporte difusivo.
* Variaciones espaciales del campo de viento.
* Emisiones.
* Altura de la capa de mezcla.
* Sedimentación de partículas.

De manera conceptual, la ecuación implementada puede escribirse como:

```text
Variación temporal
        +
Advección
        +
Efecto de divergencia del viento
        =
Difusión
        +
Fuentes
        -
Sedimentación
```

La PINN utiliza diferenciación automática para calcular:

```text
∂C/∂t

∂C/∂x
∂C/∂y

∂²C/∂x²
∂²C/∂y²
```

a partir de la propia red neuronal.

---

# Diferenciación automática

TensorFlow `GradientTape` permite obtener las derivadas de la salida del modelo con respecto a las coordenadas espaciales y temporales.

El procedimiento general es:

```text
          Cθ(x,y,t,...)
                │
       ┌────────┼────────┐
       ▼        ▼        ▼
     ∂C/∂x    ∂C/∂y    ∂C/∂t
       │        │
       ▼        ▼
    ∂²C/∂x²  ∂²C/∂y²
       │
       └────┬────┘
            ▼
      Residuo de PDE
```

El residuo de la ecuación diferencial se transforma posteriormente en una pérdida mediante el error cuadrático medio.

---

# Función de pérdida de la PINN

La PINN utiliza dos componentes principales.

## Pérdida basada en datos

```text
L_datos
```

mide la diferencia entre las concentraciones estimadas y los valores de entrenamiento.

## Pérdida física

```text
L_PDE
```

mide el grado de incumplimiento de la ecuación diferencial.

La función de pérdida utilizada durante el entrenamiento es:

```text
L_total = L_datos + 0.1 L_PDE
```

El factor `0.1` controla la contribución relativa del término físico respecto al ajuste directo de los datos.

---

# Modelo híbrido datos + física

La diferencia entre ambos modelos puede resumirse como:

```text
FNN
────────────────────────────
Entradas
   │
   ▼
Red neuronal
   │
   ▼
Predicción
   │
   ▼
Comparación con datos
   │
   ▼
L_datos


PINN
────────────────────────────
Entradas
   │
   ▼
Red neuronal
   │
   ├───────────────┐
   ▼               ▼
Predicción     Derivadas
   │               │
   ▼               ▼
L_datos         Residuo PDE
                   │
                   ▼
                 L_PDE
   │               │
   └───────┬───────┘
           ▼
 L_datos + 0.1 L_PDE
```

---

# Modelo físico de referencia

Los datos empleados durante el entrenamiento proceden de un modelo numérico de transporte y dispersión.

Por lo tanto, la comparación busca determinar qué tan adecuadamente las redes neuronales pueden reproducir la solución generada por el modelo físico.

La FNN intenta aprender directamente el mapeo entrada-salida.

La PINN recibe la misma información, pero su entrenamiento está además condicionado por la ecuación diferencial que representa el fenómeno.

---

# Evaluación espacio-temporal

El dominio utilizado durante la evaluación se representa mediante una malla bidimensional.

En el script se utiliza:

```text
24 × 24 nodos
```

con una separación aproximada de:

```text
500 m
```

entre nodos.

La concentración se actualiza únicamente en los nodos interiores y las fronteras permanecen definidas por las condiciones establecidas en la simulación.

---

# Condición inicial

La concentración inicial se construye utilizando información de una estación de monitoreo de calidad del aire.

El valor observado se asigna inicialmente a los nodos interiores:

```text
┌─────────────────┐
│ 0  0  0  0  0   │
│ 0  C  C  C  0   │
│ 0  C  C  C  0   │
│ 0  C  C  C  0   │
│ 0  0  0  0  0   │
└─────────────────┘
```

Las fronteras permanecen en cero dentro de la configuración utilizada.

---

# Variables ambientales durante la simulación

En cada paso temporal se actualizan diferentes variables necesarias para realizar la predicción.

Entre ellas:

```text
Velocidad del viento
Dirección del viento
Campo espacial de viento
Gradientes espaciales del viento
Altura de la capa de mezcla
Emisiones
Concentración previa
Concentraciones de nodos vecinos
```

Estas variables se utilizan para construir el vector de entrada de cada nodo interior.

---

# Campo de viento

El campo espacial de viento se obtiene mediante un modelo previamente entrenado.

A partir del campo calculado se estiman también:

```text
∂vx/∂x
∂vy/∂y
```

mediante pequeñas perturbaciones espaciales.

Estas cantidades forman parte de los términos físicos empleados por la PINN.

---

# Emisiones

Las emisiones se obtienen mediante un módulo independiente y se evalúan en cada paso temporal:

```text
q = q(x,y,t)
```

permitiendo representar fuentes cuya intensidad puede cambiar espacial y temporalmente.

---

# Altura de mezcla

La altura de la capa de mezcla:

```text
h(t)
```

se evalúa a partir de las condiciones meteorológicas y se incorpora como una variable relacionada con el volumen efectivo disponible para la dispersión de contaminantes.

---

# Predicción recursiva

Para cada paso temporal:

```text
1. Se evalúan las variables meteorológicas.

2. Se calcula el campo de viento.

3. Se calculan las emisiones.

4. Se obtiene la altura de mezcla.

5. Se construye el vector de entrada.

6. La red predice C(t+Δt).

7. Las concentraciones predichas sustituyen
   el estado anterior.

8. Se repite el procedimiento.
```

En términos de código, conceptualmente:

```text
C_actual
    │
    ▼
modelo(...)
    │
    ▼
C_nueva
    │
    ▼
C_actual = C_nueva
    │
    └────────► siguiente paso
```

Esto permite utilizar la red neuronal como sustituto del solucionador numérico durante la propagación temporal.

---

# Comparación FNN vs. PINN

El objetivo experimental del repositorio es comparar:

| Característica                |     FNN |    PINN |
| ----------------------------- | ------: | ------: |
| Entrenamiento con datos       |       ✓ |       ✓ |
| Restricción física            |       — |       ✓ |
| Predicción PM₂.₅              |       ✓ |       ✓ |
| Modelo no lineal              |       ✓ |       ✓ |
| Uso recursivo                 | posible | posible |
| Propagación de error          |      sí |      sí |
| Consistencia física explícita |       — |       ✓ |

La hipótesis que puede estudiarse con esta comparación es si la incorporación del conocimiento físico permite obtener un modelo más robusto cuando las predicciones deben propagarse durante múltiples pasos temporales.

---

# Métricas durante el entrenamiento

Los modelos se evalúan mediante el error cuadrático medio y el coeficiente de determinación.

Para la FNN se registra:

```text
Training MSE
Validation MSE
R²
```

Para la PINN se registran independientemente:

```text
PDE training loss
PDE validation loss

Data training loss
Data validation loss

R²
```

Esto permite analizar tanto la capacidad predictiva de la PINN como el cumplimiento de la ecuación física.

---

# Estructura del repositorio

```text
.
├── modelo_pinn.py
├── eval_modelos_fnn_pinn.py
├── entrenamiento_pinn.py
├── README.md
└── requirements.txt
```

---

## `modelo_pinn.py`

Script principal para el entrenamiento y comparación de la FNN y la PINN.

Incluye:

* Carga de la base de datos.
* Entrenamiento de una FNN convencional.
* Evaluación mediante MSE y R².
* Construcción de la PINN.
* Cálculo automático de derivadas.
* Construcción del residuo de la PDE.
* Entrenamiento con pérdida datos + física.
* Validación.
* Evaluación del ajuste de la PINN.

---

## `eval_modelos_fnn_pinn.py`

Script destinado a la evaluación espacio-temporal de los modelos mediante propagación recursiva.

Incluye:

* Construcción del dominio espacial.
* Definición de la condición inicial.
* Obtención del campo de viento.
* Evaluación de la altura de mezcla.
* Evaluación de emisiones.
* Construcción de las entradas espaciales.
* Predicción del siguiente estado.
* Realimentación de las predicciones.
* Visualización espacial de PM₂.₅.

> **Nota:** en la versión actual del script únicamente se carga explícitamente `PINN_model.h5`. Para realizar una comparación recursiva directa FNN vs. PINN debe incorporarse también la carga y propagación del modelo FNN.

---

## `entrenamiento_pinn.py`

Contiene una implementación adicional del entrenamiento de una red neuronal utilizando PyTorch.

Incluye:

* Conversión de datos a tensores.
* Escalamiento Min-Max.
* Definición de arquitectura.
* División entrenamiento-validación.
* `DataLoader`.
* Entrenamiento por lotes.
* Seguimiento de las funciones de pérdida.

Este script corresponde a una implementación/experimento adicional y no contiene la formulación física completa empleada en `modelo_pinn.py`.

---

# Archivos externos requeridos

El script de evaluación depende además de módulos y modelos desarrollados por separado, entre ellos:

```text
matriz_emisiones
modulo_alt_mezcla
modulo_velocidad_viento

modelo_viento.h5
scaler_viento.pkl
```

Estos componentes proporcionan información necesaria sobre:

* Emisiones.
* Altura de la capa de mezcla.
* Campo espacial de velocidad del viento.

Para ejecutar completamente la simulación es necesario disponer de estos archivos o adaptar el código a fuentes equivalentes.

---

# Requisitos

Las principales librerías utilizadas son:

```text
numpy
pandas
tensorflow
keras
torch
scikit-learn
matplotlib
joblib
contextily
```

Una instalación básica puede realizarse mediante:

```bash
pip install numpy pandas tensorflow keras torch scikit-learn matplotlib joblib contextily
```

---

# Orden sugerido de ejecución

```text
1. Generación de base_datos.csv
          │
          ▼
2. modelo_pinn.py
          │
          ├── FNN
          │
          └── PINN
                 │
                 ▼
3. Guardar modelos entrenados
                 │
                 ▼
4. eval_modelos_fnn_pinn.py
                 │
                 ▼
     Propagación espacio-temporal
                 │
                 ▼
      Comparación FNN vs. PINN
```

---

# Configuración

Los scripts originales contienen rutas absolutas locales.

Por ejemplo:

```python
ruta = "C:/.../"
```

Para mejorar la reproducibilidad se recomienda utilizar rutas relativas y organizar el repositorio de la siguiente manera:

```text
.
├── data/
│   └── base_datos.csv
│
├── models/
│   ├── FNN_model.h5
│   ├── PINN_model.h5
│   ├── modelo_viento.h5
│   └── scaler_viento.pkl
│
├── src/
│   ├── modelo_pinn.py
│   ├── eval_modelos_fnn_pinn.py
│   ├── entrenamiento_pinn.py
│   ├── matriz_emisiones.py
│   ├── modulo_alt_mezcla.py
│   └── modulo_velocidad_viento.py
│
├── results/
│   └── figures/
│
├── requirements.txt
└── README.md
```

---

# Interpretación del estudio

El interés principal del enfoque no es únicamente determinar cuál modelo presenta menor error durante el entrenamiento.

La comparación cobra especial importancia cuando los modelos se utilizan como **modelos dinámicos recursivos**.

Una red puede presentar un buen ajuste de un solo paso:

```text
C(t) ──► C(t+Δt)
```

pero producir resultados inestables cuando se utiliza repetidamente:

```text
C(t)
  │
  ▼
Ĉ(t+Δt)
  │
  ▼
Ĉ(t+2Δt)
  │
  ▼
Ĉ(t+3Δt)
  │
  ▼
 ...
```

En esta situación, los errores de predicción pueden acumularse y modificar progresivamente el estado empleado como entrada.

El uso de una PINN permite estudiar si añadir información física al entrenamiento contribuye a limitar esta propagación del error y genera trayectorias espacio-temporales más consistentes con los mecanismos de transporte y dispersión atmosférica.

---

# Aplicación

Los códigos fueron desarrollados para el modelado de la distribución espacio-temporal de **PM₂.₅ en Aguascalientes, México**.

El enfoque combina:

```text
Datos
+
Modelo numérico
+
Redes neuronales
+
Ecuaciones diferenciales
+
Predicción recursiva
```

con el objetivo de estudiar modelos sustitutos (*surrogate models*) capaces de aproximar la dinámica de un modelo físico de transporte y dispersión.

---

# Autor

**Héctor Antonio Olmos Guerrero**

Research in environmental modeling, air quality and machine learning.

---

## Licencia

Este repositorio se proporciona con fines académicos y de investigación.

La licencia específica puede definirse de acuerdo con las condiciones deseadas para el uso, modificación y distribución del código.
