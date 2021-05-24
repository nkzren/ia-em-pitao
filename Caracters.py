from sklearn.neural_network import MLPClassifier
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

caracteres_treino = pd.read_csv("caracteres-limpo.csv", header=None)
caracteres_teste = pd.read_csv("caracteres-ruido.csv", header=None)

colunas_rotulos = [63, 64, 65, 66, 67, 68, 69]
rotulos = caracteres_treino.filter(colunas_rotulos, axis = 1)
rotulos_array = rotulos.values

X = caracteres_treino.drop(colunas_rotulos,axis = 1)
#print(X.shape)
y = rotulos
#print(y.shape)

classifier = MLPClassifier(hidden_layer_sizes=(6), max_iter=30000,activation = 'relu',solver='adam',random_state=1)
classifier.fit(X, y)

testes = caracteres_teste.drop(colunas_rotulos,axis = 1)
respostas_testes = caracteres_teste.filter(colunas_rotulos, axis = 1)

y_pred = classifier.predict(testes)

number = r2_score(y_pred, respostas_testes)

print("Acurácia da IA: %.16f " % number)

#print(y_pred)

#print(respostas_testes)

