import numpy as np # a grande biblioteca salvadora de todas as ias
import pandas as pd #somente para aceitar pandas pra devolver para numpy
import FuncoesAtivacao as func #funções de ativação para qualquer coisa n

class MultiPerceptron:

    # https://towardsdatascience.com/softmax-regression-in-python-multi-class-classification-3cb560d90cb2

    # https://colab.research.google.com/drive/1Y1Id58f1OjWd-bauF1MZccGDW_pUYU9n?usp=sharing

    # Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
    # .\.venv\Scripts\activate

    def __init__(self, hiddenNodes, taxa, epocas, funcao, parada):
        self.hiddenNodes = hiddenNodes        
        self.epocas = epocas
               
        self.pesoEntradaMeio = None
        self.pesoMeioSaida = None
        self.min = 0
        self.max = 0
        self.parada = parada or False
        
        self.taxa = taxa or 0.1


        if(funcao == 'sigmoid'):
            self.funcao = func.FuncoesAtivacao.sigmoid
            self.derivativa = func.FuncoesAtivacao.dsigmoid
            self.min = func.FuncoesAtivacao.minsigmoid
            self.max = func.FuncoesAtivacao.maxsigmoid
        #efif(funcao == 'relu'):
            #self.funcao = func.FuncoesAtivacao.relu
            #self.derivativa = func.FuncoesAtivacao.drelu        
        #elif(funcao == 'tanh'):
            #self.funcao = func.FuncoesAtivacao.tanh
            #self.derivativa = func.FuncoesAtivacao.dtanh 
        #elif(funcao == 'softmax'): #não use o softmax, ô derivada desgraçada
            #self.funcao = func.FuncoesAtivacao.softmax
            #self.derivativa = func.FuncoesAtivacao.dsoftmax
        else:
            raise Exception('Função de ativação não suportada')
        
        
    #@vectorize(["float32(float32, float32)"], target='cuda')  se o numba funcionasse, claro.
    def train(self,X,y):            
       
        #transformações de dataframe do pandas pra array do numpy
        if isinstance(X, pd.DataFrame):
            inputM = X.to_numpy()
        else:
            inputM = X
        
        if isinstance(y, pd.DataFrame):
            outputM = y.to_numpy()
        else:
            outputM = y

        #dimensões das entradas

        m, n = inputM.shape  # aqui temos m = número de entradas, n = número de *features*

        lin_2, col_2 = outputM.shape  #lin_2 = número de respostas, col_2 = número de *labels*    

        #inicializar as matrizes de peso
        self.pesoEntradaMeio = np.random.uniform(self.min, self.max,(n,self.hiddenNodes))
        
        self.pesoMeioSaida = np.random.uniform(self.min, self.max,(self.hiddenNodes,col_2))

        #self.vies_1 = np.random.rand(n)
        #self.vies_2 = np.random.rand(self.hiddenNodes)

        custo_anterior = 100 # necessário para a verificação de convergência abaixo
                             # precisa dar muito errado para que (isso) menos (custo depois de 20k épocas) dê entre 0 e 0.001
                
        contador_de_epocas = 0
        # isso faz entrada x pesos da camada escondida, depois 
        #  a saida disso x pesos da camada de saida
        # estou muito feliz que isso dá certo eu demorei 2 dias pra entender
        while(contador_de_epocas <= self.epocas):
            #forward step:
            inputHidden = np.dot(inputM, self.pesoEntradaMeio)
            
            outputHidden = self.funcao(inputHidden)

            inputSaida = np.dot(outputHidden, self.pesoMeioSaida)            
           
            outputSaida = self.funcao(inputSaida)

            #função de custo:
            custo = (-1/m) * np.sum(outputM * np.log(outputSaida) + (1 - outputM) * np.log(1 - outputSaida ))
        
            #----------------------------------------------------------------------------
            #HORA DO BACKPROPAGATION
            #calculando o gradiente de [hidden -> saida] :  Y-Y^
            erro_saida = np.subtract(outputM, outputSaida) # error
            
            gradienteOutput = np.multiply(erro_saida, self.derivativa(outputSaida))  #GRADIENTE CAMADA X \ erro * derivada(y^)

            #"recursivamente" calculo o peso da próxima camada:

            #calculando o gradiente de [entrada -> hidden]  GRADIENTE DA CAMADA X-1 
            erro_hidden = gradienteOutput.dot(self.pesoMeioSaida.T) # error_1 = 

            gradienteHidden = np.multiply(erro_hidden, self.derivativa(outputHidden))

            #ajustando os pesos da camada entrada -> hidden
            self.pesoEntradaMeio += inputM.T.dot(gradienteHidden) * self.taxa

            #ajustando os pesos da camada hidden -> saida
            self.pesoMeioSaida += outputHidden.T.dot(gradienteOutput) * self.taxa

            #o meu método de parada antecipada, checa o quanto a função de custo convergiu a cada 20 mil iterações.
            contador_de_epocas += 1            
            if(contador_de_epocas % 20000 == 0 and self.parada == True):
                if(custo_anterior - custo < 0.1 and custo_anterior - custo > 0):
                    print("O custo convergiu abaixo de .1, saíndo.")
                    break 
                custo_anterior = custo
                print('Além de %d epocas.' % contador_de_epocas)
                print('Custo = %.16f' % custo)
                
        
    def query(self, X):

        # é basicamente o forwardstep. os dados passam 1 vez e retornam a imagem.
        
        tabela = X

        if isinstance(tabela, pd.DataFrame):
                inputM = tabela.to_numpy()
        else:
            inputM = tabela
            
        inputHidden = np.dot(inputM, self.pesoEntradaMeio)            
        outputHidden = self.funcao(inputHidden)
        inputSaida = np.dot(outputHidden, self.pesoMeioSaida)            
        outputSaida = self.funcao(inputSaida)

        return outputSaida
