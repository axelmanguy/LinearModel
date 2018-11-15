
#exemple inspire de http://scikit-learn.org/stable/_downloads/plot_isotonic_regression.py

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#PARTIE 1 : Utilisation de scikit-learn pour la regression lineaire
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#generation de donnees test
n = 100
x = np.arange(n)
y = np.random.randn(n)*30 + 50. * np.log(1 + np.arange(n))

# instanciation de sklearn.linear_model.LinearRegression
lr = LinearRegression()
lr.fit(x[:, np.newaxis], y)  # np.newaxis est utilise car x doit etre une matrice 2d avec 'LinearRegression'

# representation du resultat
fig = plt.figure()
plt.plot(x, y, 'r.')
plt.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
plt.legend(('Data', 'Linear Fit'), loc='lower right')
plt.title('Linear regression')
plt.show()


#QUESTION 1.1 : 
#Bien comprendre le fonctionnement de lr, en particulier lr.fit et lr.predict

#QUESTION 1.2 :
#On s'interesse a x=105. En supposant que le model lineaire soit toujours 
#valide pour ce x, quelles valeur corresondante de y vous semble la plus 
#vraisemblable ? 


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#PARTIE 2 : impact et detection d'outliers
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#generation de donnees test
n = 10
x = np.arange(n)
y = 10. + 4.*x + np.random.randn(n)*3. 
y[9]=y[9]+20

# instanciation de sklearn.linear_model.LinearRegression
lr = LinearRegression()
lr.fit(x[:, np.newaxis], y)  # np.newaxis est utilise car x doit etre une matrice 2d avec 'LinearRegression'

# representation du resultat

print('b_0='+str(lr.intercept_)+' et b_1='+str(lr.coef_[0]))

fig = plt.figure()
plt.plot(x, y, 'r.')
plt.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
plt.legend(('Data', 'Linear Fit'), loc='lower right')
plt.title('Linear regression')
plt.show()


#QUESTION 2.1 : 
#La ligne 'y[9]=y[9]+20' genere artificiellement une donnee aberrante.
#-> Tester l'impact de la donnee aberrante en estimant b_0, b_1 et s^2 
#   sur 5 jeux de donnees qui la contiennent cette donnee et 5 autres qui
#   ne la contiennent pas (simplement ne pas executer la ligne y[9]=y[9]+20).
#   On remarque que $\beta_0 = 10$, $\beta_1 = 4$ et $sigma=3$ dans les 
#   données simulees.


#QUESTION 2.2 : 
#2.2.a -> Pour chaque variable i, calculez les profils des résidus 
#         $e_{(i)j}=y_j - \hat{y_{(i)j}}$ pour tous les j, ou   
#         \hat{y_{(i)j}} est l'estimation de y_j a partir d'un modele  
#         lineaire appris sans l'observation i.
#2.2.b -> En quoi le profil des e_{(i)j} est different pour i=9 que pour  
#         les autre i
#2.2.c -> Etendre ces calculs pour définir la distance de Cook de chaque 
#         variable i
#
#AIDE : pour enlever un element 'i' de 'x' ou 'y', utiliser 
#       x_del_i=np.delete(x,i) et y_del_i=np.delete(y,i) 




