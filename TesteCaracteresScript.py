import pandas as pd # usado para consertar os dados (porque tava triste)
import numpy as np # usado porque arrays.
from sklearn.metrics import r2_score #usado para avaliação, depois eu justifico
import Multilayer_Perceptron_Classificador as mlpc #óbvio

caracteres_treino = pd.read_csv("caracteres-limpo.csv", header=None)
caracteres_teste = pd.read_csv("caracteres-ruido.csv", header=None)

caracteres_teste = caracteres_teste.sample(frac=1)

colunas_rotulos = [63, 64, 65, 66, 67, 68, 69]
rotulos = caracteres_treino.filter(colunas_rotulos, axis = 1)
rotulos_array = rotulos.values

X = caracteres_treino.drop(colunas_rotulos,axis = 1)
y = rotulos

testes = caracteres_teste.drop(colunas_rotulos,axis = 1)
respostas_testes = caracteres_teste.filter(colunas_rotulos, axis = 1)

mlp = mlpc.MultiPerceptron(90, 0.1, 300000, 'sigmoid') # nós_escondidos, taxa de aprendizado, épocas_maximas, função de ativação. nessa ordem

mlp.train(X, y) #para este método, passe apenas X = tabela de features e y = tabela de targets, nessa ordem.

y_pred = mlp.query(testes)

number = r2_score(y_pred, respostas_testes)

#y_pred = y_pred.round(decimals=3)

#y_pred_df = pd.DataFrame(y_pred)

#print(y_pred_df)

print("Acurácia da IA: %.16f " % number)





