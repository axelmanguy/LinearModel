
#exemple inspire de http://scikit-learn.org/stable/_downloads/plot_isotonic_regression.py

import numpy as np
import matplotlib.pyplot as plt





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

#-> lr est un objet qui permet d'effectuer la regression lineaire
#-> lr.fit permet d'apprendre les parametres du modele a partir de donnees d'apprentissage
#-> lr.predict permet de predire un 'y' partir d'un 'x' test

#QUESTION 1.2 :
#On s'interesse a x=105. En supposant que le model lineaire soit toujours 
#valide pour ce x, quelles valeur corresondante de y vous semble la plus 
#vraisemblable ? 

lr.predict([[105]])

#la valeur est 264.80754151
#On remarque que les valeurs donnees pour la prediction doivent etre dans un vecteur colonne, ici une matrice 1x1

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

#sans donnee aberrante
for i in range(5):
  n = 10
  x = np.arange(n)
  y = 10. + 4.*x + np.random.randn(n)*3. 
  lr.fit(x[:, np.newaxis], y)  # np.newaxis est utilise car x doit etre une matrice 2d avec 'LinearRegression'
  print('b_0='+str(lr.intercept_)+' b_1='+str(lr.coef_[0])+' / les valeurs recherchees sont 10 et 4')
  s=np.std(y-lr.predict(x[:, np.newaxis]))
  print('Bruit estime='+str(s)+'    /  reel = 3 ')

#avec donnee aberrante
for i in range(5):
  n = 10
  x = np.arange(n)
  y = 10. + 4.*x + np.random.randn(n)*3. 
  y[9]=y[9]+20
  lr.fit(x[:, np.newaxis], y)  # np.newaxis est utilise car x doit etre une matrice 2d avec 'LinearRegression'
  print('b_0='+str(lr.intercept_)+' b_1='+str(lr.coef_[0])+' / les valeurs recherchees sont 10 et 4')
  s=np.std(y-lr.predict(x[:, np.newaxis]))
  print('Bruit estime='+str(s)+'    /  reel = 3 ')


#-> estimations correctes sans donnee aberrante mais biaisees avec la donnee aberrante.
#   On peut mesurer le biais en comparant la moyenne des parametres estimes aux vrais valeures des parametres
#   sur un grand nombre de repetitions

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

#-> regeneration de donnees biaisees 
n = 10
x = np.arange(n)
y = 10. + 4.*x + np.random.randn(n)*3. 
y[9]=y[9]+20

#-> question 2.2.a
for i in range(n):
  x_del_i=np.delete(x,i)
  y_del_i=np.delete(y,i) 
  
  lr.fit(x_del_i[:, np.newaxis], y_del_i)
  
  print('variable supprimee='+str(i))
  print('residus ='+str(y-lr.predict(x[:, np.newaxis])))

#2.2.b -> l'estimation de j=9 est toujours la plus mauvaise. Elle est plus mauvaise quand i=9  
#         enleve de l'apprentissage que pour tous les autres i est, alors que toutes les 
#         autres predictions sont meilleures.
#         ... on peut clairement se douter que cette observation est un outlier (donnee aberrante). 

#-> question 2.2.c

lr.fit(x[:, np.newaxis], y)  # np.newaxis est utilise car x doit etre une matrice 2d avec 'LinearRegression'
s2=np.sum((y-lr.predict(x[:, np.newaxis]))*(y-lr.predict(x[:, np.newaxis])))/(n-2)
          (y-lr.predict(x[:, np.newaxis]))*(y-lr.predict(x[:, np.newaxis])))

for i in range(n):
  x_del_i=np.delete(x,i)
  y_del_i=np.delete(y,i) 
  
  lr.fit(x_del_i[:, np.newaxis], y_del_i)
  sum_squared_error=np.sum((y-lr.predict(x[:, np.newaxis]))*(y-lr.predict(x[:, np.newaxis])))
  
  print('D('+str(i)+')='+str(sum_squared_error/(2.*s2)))

#->la distance est plus grande pour la valeur 9 que toutes les autres qui sont stable. La donnee 
#  aberrante est encore retrouvee.

