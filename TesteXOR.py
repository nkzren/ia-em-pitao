import pandas as pd # usado para consertar os dados (porque tava triste)
import numpy as np # usado porque arrays.
from sklearn.metrics import r2_score #usado para avaliação, depois eu justifico
import Multilayer_Perceptron_Classificador as mlpc #óbvio

xor = pd.read_csv("problemXOR.csv", header=None)

X = xor.drop([2], axis=1).drop([3], axis=0)
y = xor.filter([2], axis=1).drop([3], axis=0)

teste_x = xor.drop([2], axis=1).drop([0,1,2], axis=0)
teste_y = xor.filter([2], axis=1).drop([0,1,2], axis=0)

#a imagem é [0, 1], portanto uma função que se encaixa é a de sigmoid

mlp = mlp = mlpc.MultiPerceptron(60, 2, 300000, 'sigmoid')

mlp.train(X,y)

y_pred = mlp.query(X)

number = r2_score(y_pred, y)

y_pred = y_pred.round(decimals=3)
y_pred_df = pd.DataFrame(y_pred)

print(y_pred_df)

print("Acurácia da IA: %.16f " % number)