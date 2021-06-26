import pandas as pd # usado para consertar os dados (porque tava triste)
import numpy as np # usado porque arrays.
from sklearn.metrics import r2_score #usado para marcar a acurácia
import matplotlib.pyplot as plt # plot de matriz de confusão parte 1
import seaborn as sn # plot de matriz de confusão parte 2
import Multilayer_Perceptron_Classificador as mlpc #óbvio
import copy


def matrizDeConfusao(y_true, y_pred):
    
    valor_x, valor_y = y_true.shape        

    resposta = np.zeros((valor_y,valor_y))    

    for i in range(valor_x):
        for j in range(valor_y):
            if y_pred[i][j] >= 0.5:
                if y_true[i][j] == 1:
                    resposta[j][j] += 1
                else:
                    for k in range(valor_y):
                        if y_true[i][k] == 1:
                            resposta[j][k] += 1
    
    return resposta


#---------------------------------------------------------------------------------- #importe os dados


caracteres_treino = pd.read_csv("caracteres-limpo.csv", header=None) #importe os dados
caracteres_teste = pd.read_csv("caracteres-ruido.csv", header=None)
caracteres_validacao = pd.read_csv("caracteres_ruido20.csv", header=None)

caracteres_teste = caracteres_teste.sample(frac=1)
caracteres_validacao = caracteres_validacao.sample(frac=1)

colunas_rotulos = [63, 64, 65, 66, 67, 68, 69] #consertando a falta de formatação dos csvs

X = caracteres_treino.drop(colunas_rotulos,axis = 1)    #conjunto de treino
y =caracteres_treino.filter(colunas_rotulos, axis = 1)                                             #repostas do conjunto de treino

valid = caracteres_validacao.drop(colunas_rotulos,axis = 1)
valid_respostas = caracteres_validacao.filter(colunas_rotulos, axis = 1)

testes = caracteres_teste.drop(colunas_rotulos,axis = 1)                #conjunto de testes
respostas_testes = caracteres_teste.filter(colunas_rotulos, axis = 1)   # respostas do conjunto de testes

#----------------------------------------------------------------------------------  TREINE O BAGULHO

# Como será feito: cinco MLPs serão iniciados, dentre os cinco, aquele que se sair melhor com o
# set de validação será eleito, e ele será utilizado no set de teste.

parametros_base = mlpc.MultiPerceptron(17, 0.1, 850, 'sigmoid', False)

mlp1 = copy.copy(parametros_base)
mlp2 = copy.copy(parametros_base)
mlp3 = copy.copy(parametros_base)
mlp4 = copy.copy(parametros_base)
mlp5 = copy.copy(parametros_base)

modelos = [mlp1, mlp2, mlp3, mlp4, mlp5]

mlp1.train(X,y)
mlp2.train(X,y)
mlp3.train(X,y)
mlp4.train(X,y)
mlp5.train(X,y)

verificacao = [-1.0]

for model in modelos:
    resultado = model.query(valid)
    number = r2_score(resultado, valid_respostas)
    verificacao.append(number)

if max(verificacao) == -1.0:
    print("deu ruim")
else :

    for numero in verificacao:
        print(numero)

    indice_eleito = verificacao.index(max(verificacao))
    print("o indice eleito foi: %d." % indice_eleito)
    eleito = modelos[indice_eleito]

#---------------------------------------------------------------------------------- TESTE DE VERDADE

y_pred = eleito.query(testes)
number = r2_score(y_pred, respostas_testes)
print("Acurácia da IA: %.16f " % number)