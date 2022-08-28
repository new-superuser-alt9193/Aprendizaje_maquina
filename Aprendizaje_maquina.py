import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# El objetivo de este dataset es encontrar una regresion lineal que permita predecir el 
# peso en gramos de un pez en base a sus carateristicas.
df = pd.read_csv('./Fish.csv') #https://www.kaggle.com/code/nitinchoudhary012/fish-weight-prediction/data

# Una vez cargado los datos usando dataprep analice las correlaciones entre los datos, 
# y para asegurarme de mis observaciones utilice seaborn para graficar los datos en base a la especie.

# En este codigo no esta presente dataprep, ya que al ejecutarlo en mi maquina en un ambiente diferente a
# un notebook terminaba por provocar que dejara de funcionar el codigo.

plt.figure(figsize=(15,6))
sns.pairplot(data= df,
             x_vars = ['Length1','Length2','Length3','Height','Width'],
             y_vars = 'Weight', 
             hue = 'Species')
plt.show()

# Eliminamos las columnas Length2 y Length3 por tener una alta correlacion con Length1
df = df.drop(columns = ['Length2', 'Length3'])
# Remplazamos los valores categoricos de las especies por valores numericos.
df['Species'] = df['Species'].replace(['Perch','Bream','Roach','Pike','Smelt','Parkki','Whitefish'],[1,2,3,4,5,6,7])

# Guardamos la limpieza y transformacion de los datos.
df.to_csv('clean_fish.csv')

# Segmentamos nuestras variables en variables depedientes e indepedientes.
y = df['Weight'].to_numpy()
# Obtenemos el numero de muestras presentes.
m = len(y)
x = df.drop(columns=['Weight'])
# Agregamos una columna extra a las variables idepediente para poder calcular el valor de b de la pendiente
x = np.c_[np.ones((len(x),1)), x]

# Generamos coeficientes aleatorios para nuestras variables indepedientes.
teta = np.random.randn(5)

# Funcion para calcular el costo de una regresion lineal multivariable
# 1/2m * sumatoria((y_hat - y)**2)
def Costo(x, y, teta):
    y_hat = x.dot(teta)
    errores = np.subtract(y_hat, y)
    # J = costo de la regresion
    J = np.sum(np.square(errores)) / (2*m)
    return J

def GradienteDescediente(x, y, teta, alfa, epocas):
    historialCostos = np.zeros(epocas)
    
    for i in range(epocas):
        y_hat = x.dot(teta)
        error =np.subtract(y_hat, y)
        delta = (2/m) * x.transpose().dot(error)
        teta = teta - alfa * delta
        historialCostos[i] = Costo(x,y,teta)

    return teta, historialCostos

# Definimos nuestro learning rate (alfa) y el numero de iteraciones a realizar
alfa = .00001
epocas = 100

teta, historialCostos = GradienteDescediente(x, y, teta, alfa, epocas)

print(teta)
print((historialCostos[-1]))

# En la siguiente grafica podemos observar el como a medida que pasan las iteraciones el error disminuye
plt.plot(range(1, epocas + 1), historialCostos, color = 'blue')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel("Epocas")
plt.ylabel("Costo (J)")
plt.show()

error = 1
while np.sqrt(np.square(error) > 0.001):
    teta, historialCostos = GradienteDescediente(x, y, teta, alfa, epocas)
    y_hat = x.dot(teta)
    error =np.mean(np.subtract(y, y_hat))
    print(np.sqrt(np.square(error)))

print(historialCostos[-1], teta)