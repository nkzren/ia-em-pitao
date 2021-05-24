import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# py -3 -m venv .venv < ------- rode esse apenas 1 vez

# .venv\scripts\activate <------- roda esse 

iris = datasets.load_iris()

X = iris.data
y = iris.target

escalador = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

padronizado = escalador.fit_transform(X_train)

#print(padronizado)

X_train_std = escalador.transform(X_train)

X_test_std = escalador.transform(X_test)

ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)

ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)

print(y_pred)
print(y_test)

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

#print(y[:5])
#print (X[:5])