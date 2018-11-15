#Exercice inspiré de http://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_compare_cross_decomposition.html

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# regression PLS avec reponse univariee (PLS1)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#Generation de donnees
n = 1000
p = 10
X = np.random.normal(size=n * p).reshape((n, p))
y = X[:, 0] + 2 * X[:, 1] + np.random.normal(size=n) + 5

#QUESTION 1: Comment sont construites les donnees simulees ? 
#            Que vous attendez-vous a voir dans la PLS

pls1 = PLSRegression(n_components=5)
pls1.fit(X, y)

print(np.round(pls1.coef_, 1))

#QUESTION 2: Quelle est la signification des pls1.coef_ et correspondent
#            -ils a ce que vous attendiez ?
#            Que renvoi de plus 'pls1.predict(X)' ? Comparez ce resultat 
#            a y.

#QUESTION 3: Est-ce qu'une regression-multiple avec selection de modele
#            conduirait a des resultats similaires ?


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# regression PLS avec reponse multivariee (PLS2)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

n = 1000
q = 3
p = 10
X = np.random.normal(size=n * p).reshape((n, p))
B = np.array([[1, 2] + [0] * (p - 2)] * q).T
Y = np.dot(X, B) + np.random.normal(size=n * q).reshape((n, q)) + 5

#QUESTION 4: Comment sont construites les donnees simulees ? 

pls2 = PLSRegression(n_components=5)
pls2.fit(X, Y)

#QUESTION 5: Quelle est l'estimation de B par le modele ? Vous semble-t-elle
#            bonne ?

