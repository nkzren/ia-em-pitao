import numpy as np

class FuncoesAtivacao:

    maxsigmoid = 1
    minsigmoid = -1
    
    def relu(x):
        return np.maximum(x,0)

    def drelu(x):
        x[x<=0] = 0
        x[x>0] = 1
        return x

    def sigmoid(x):
        return 1/ (1 + np.exp(-x))

    def dsigmoid(x):
        return x * (1-x)
    
    def tanh(x):
        return np.tanh(x)         

    def dtanh(x):
        temp = np.cosh(x)
        return 1/np.square(temp)

    def softmax(x):        
        exp = np.exp(x - np.max(x))
                        
        return exp / exp.sum(keepdims=True, axis=1)
    
    def dsoftmax(x):            
        m,n = x.shape 
        da = FuncoesAtivacao.dsigmoid(x)      
        tensor1 = np.einsum('ij,ik->ijk', x, x)
        tensor2 = np.einsum('ij,jk->ijk', x, np.eye(n, n))
        dSoftmax = tensor2 - tensor1        
        return np.einsum('ijk,ik->ij', dSoftmax, da)

    
