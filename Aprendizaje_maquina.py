import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./Fish.csv')

plt.figure(figsize=(15,6))
sns.pairplot(data= df,
             x_vars = ['Length1','Length2','Length3','Height','Width'],
             y_vars = 'Weight', 
             hue = 'Species')
plt.show()

df = df.drop(columns = ['Length2', 'Length3'])
df['Species'] = df['Species'].replace(['Perch','Bream','Roach','Pike','Smelt','Parkki','Whitefish'],[1,2,3,4,5,6,7])

df.to_csv('clean_fish.csv')

y = df['Weight'].to_numpy()
m = len(y)
x = df.drop(columns=['Weight'])
x = np.c_[np.ones((len(x),1)), x]

teta = np.random.randn(5,1)

# Funcion para calcular el costo de una regresion lineal multivariable
# 1/2m * sumatoria((y_hat - y)**2)
def Costo(x, y, teta):
    y_hat = x.dot(teta)
    errores = np.subtract(y_hat, y)
    # J = costo de la regresion
    J = 1 / (2 * m) * np.sum(np.square(errores))
    return J

def GradienteDescediente(x, y, teta, alfa, epocas):
    historialCostos = np.zeros(epocas)

    for i in range(epocas):
        y_hat = x.dot(teta)
        error = np.subtract(y_hat,y)
        #gradienteCosto = x.transpose().dot(error)
        delta = (alfa/m) * x.transpose().dot(error)
        teta = teta - delta #- alfa 
        historialCostos[i] = Costo(x,y,teta)

    return teta[4], historialCostos

def GradienteDescediente(X, y, theta, alpha, iterations):
  """
  Compute cost for linear regression.

  Input Parameters
  ----------------
  X : 2D array where each row represent the training example and each column represent the feature ndarray. Dimension(m x n)
      m= number of training examples
      n= number of features (including X_0 column of ones)
  y : 1D array of labels/target value for each traing example. dimension(m x 1)
  theta : 1D array of fitting parameters or weights. Dimension (1 x n)
  alpha : Learning rate. Scalar value
  iterations: No of iterations. Scalar value. 

  Output Parameters
  -----------------
  theta : Final Value. 1D array of fitting parameters or weights. Dimension (1 x n)
  cost_history: Conatins value of cost for each iteration. 1D array. Dimansion(m x 1)
  """
  cost_history = np.zeros(iterations)

  for i in range(iterations):
    predictions = X.dot(theta)
    errors = np.subtract(predictions, y)
    sum_delta = (alpha / m) * X.transpose().dot(errors)
    theta = theta - sum_delta
    cost_history[i] = Costo(X, y, theta)  

  return theta, cost_history

#alfa = float(input("Learning rate:"))
#epocas = int(input("Epocas: "))
alfa = .001
epocas = 10000

teta, historialCostos = GradienteDescediente(x, y, teta, alfa, epocas)

print(teta)

plt.plot(range(1, epocas + 1), historialCostos, color = 'blue')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel("Epocas")
plt.ylabel("Costo (J)")
plt.show()
