import pandas as pd # usado para consertar os dados (porque tava triste)
import numpy as np # usado porque arrays.
from sklearn.metrics import r2_score #usado para marcar a acurácia
import matplotlib.pyplot as plt # plot de matriz de confusão parte 1
import seaborn as sn # plot de matriz de confusão parte 2
import Multilayer_Perceptron_Classificador as mlpc #óbvio


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

caracteres_teste = caracteres_teste.sample(frac=1)

colunas_rotulos = [63, 64, 65, 66, 67, 68, 69] #consertando a falta de formatação dos csvs
rotulos = caracteres_treino.filter(colunas_rotulos, axis = 1)
rotulos_array = rotulos.values

X = caracteres_treino.drop(colunas_rotulos,axis = 1)    #conjunto de treino
y = rotulos                                             #repostas do conjunto de treino

testes = caracteres_teste.drop(colunas_rotulos,axis = 1)                #conjunto de testes
respostas_testes = caracteres_teste.filter(colunas_rotulos, axis = 1)   # respostas do conjunto de testes

#----------------------------------------------------------------------------------  TREINE O BAGULHO

mlp = mlpc.MultiPerceptron(17, 0.1, 10000, 'sigmoid', False) # nós_escondidos, taxa de aprendizado, épocas_maximas, função de ativação. nessa ordem

mlp.train(X, y) #para este método, passe apenas X = tabela de features e y = tabela de targets, nessa ordem.

y_pred = mlp.query(testes)

number = r2_score(y_pred, respostas_testes)

#---------------------------------------------------------------------------------- PRINT DA RESPOSTA (debug)

y_pred = y_pred.round(decimals=2) # isso deixa a matriz resultante mais legivel
y_pred_df = pd.DataFrame(y_pred)
print(y_pred_df)
print(respostas_testes)
#---------------------------------------------------------------------------------- PRINT DA MATRIZ DE CONFUSÃO
cm = matrizDeConfusao(respostas_testes.to_numpy(), y_pred)
sn.heatmap(cm, annot=True, xticklabels=('A','B','C','D','E','F','K'), yticklabels=('A','B','C','D','E','F','K'))
plt.xlabel("y_true") 
plt.ylabel("y_pred") 
plt.show()
#---------------------------------------------------------------------------------- 

print("Acurácia da IA: %.16f " % number)









