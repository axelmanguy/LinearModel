import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

#1) READ AND EXTRACT DATA

#1.1) get data out of the csv file
dataframe=pandas.read_csv("./RealMedicalData.csv",sep=';',decimal=b',')

listColNames=list(dataframe.columns)


#1.2) extract X and Y as numpy arrays

XY=dataframe.values
ColNb_Y=listColNames.index('Disease progression')


Y=XY[:,ColNb_Y].reshape((XY.shape[0],1))   #reshape is to make sure that Y is a column vector
X = np.delete(XY, ColNb_Y, 1)

X_scaled = preprocessing.scale(X)

listColNames.pop(ColNb_Y)     #to make it contains the column names of X only

#2) EXPLORE THE DATA

for Col in range(len(listColNames)):
  plt.plot(X[:,Col],Y[:],'.')
  plt.xlabel(listColNames[Col])
  plt.ylabel('Disease progression')
  plt.show()

#3) PERFORM THE REGRESSION

#3.1) ridge regression
from sklearn.linear_model import Ridge

ridge_regressor=Ridge(alpha=1.0, fit_intercept=True)

ridge_regressor.fit(X_scaled,Y)

print('Beta values')
for Col in range(len(listColNames)):
  print('-> '+listColNames[Col]+': '+str(ridge_regressor.coef_[0,Col]))

#3.2) Lasso regression

from sklearn.linear_model import Lasso

lasso_regressor=Lasso(alpha=0.5, fit_intercept=True)

lasso_regressor.fit(X_scaled,Y)

print('Beta values')
for Col in range(len(listColNames)):
  print('-> '+listColNames[Col]+': '+str(lasso_regressor.coef_[Col]))





#QUESTION 1: Ouvrir le fichier RealMedicalData.csv (par exemple avec libreoffice) puis comprendre 
#            chaque instruction du code

#-> on remarque que la variable 'Acid 1 density' est la seule qui semble avoir une relation lineaire viable
#   avec  'Disease progression' 

#QUESTION 2: Trouver une bonne valeur de alpha pour la regression Ridge et Lasso en utilisant la
#            separation des observations en un jeu d'apprentissage et un jeu de validation
#              -> Est-ce que les deux méthodes ont un bon pouvoir de prediction (utiliser metrics.r2_score) ?

#-> REMARQUE: VU QUE LES DONNEES SONT TRES DIFFICILES, ON UTILISE UN LEAVE-ONE-OUT PLUTOT QU'UN K-FOLD ET 
#             ON MESURE LA SOMME DES ERREURS D'APPROXIMATION AU CARRE
#-> REMARQUE 2: DEUX OUTLIERS DETECTES VISUELLEMENT ONT ETE ENLEVEES DU 1ER JEU DE DONNEE DU TP (LES OUTLIERS
#               SONT ENLEVES DANS LA VERSION ACTUELLEMENT DISTRIBUEE)

#-> 2.1 avec lasso
from sklearn.model_selection import LeaveOneOut

Loo=LeaveOneOut()
for alpha in [0.01,0.1,0.2,0.4,0.5,0.6,0.7,0.8,1.,10.]:
  sum_squared_scores=0.
  for train, test in Loo.split(X):
    #print(train, test)
    lasso_regressor = Lasso(alpha=alpha, fit_intercept=True)
    lasso_regressor.fit(X_scaled[train], Y[train])
  
    y_pred_lasso = lasso_regressor.predict(X_scaled[test])
    #print(Y[test], y_pred_lasso)
    sum_squared_scores+=(Y[test][0][0]- y_pred_lasso[0])*(Y[test][0][0]- y_pred_lasso[0])
  print(alpha," total: ",sum_squared_scores)

#-> meilleur score avec alpha=0.6

lasso_regressor = Lasso(alpha=0.6, fit_intercept=True)
lasso_regressor.fit(X_scaled, Y)
print(lasso_regressor.coef_)

#-> On remarque que la 1ere colonne ('Acid 1 density') a ete la seule a etre selectionnee. Elle corresponds en 
#   fait a la seule donnee medicale liee a 'Disease progression' 

#-> 2.2 avec Ridge

Loo=LeaveOneOut()
for alpha in [0.01,0.1,1.,10.,50.,75.,100.,250.,500.,1000.]:
  sum_squared_scores=0.
  for train, test in Loo.split(X):
    #print(train, test)
    ridge_regressor=Ridge(alpha=alpha, fit_intercept=True)
    ridge_regressor.fit(X_scaled[train], Y[train])
  
    y_pred_ridge = ridge_regressor.predict(X_scaled[test])
    #print(Y[test], y_pred_lasso)
    sum_squared_scores+=(Y[test][0][0]- y_pred_ridge[0])*(Y[test][0][0]- y_pred_ridge[0])
  print(alpha," total: ",sum_squared_scores)

#-> meilleur score avec alpha=100.0

ridge_regressor=Ridge(alpha=100., fit_intercept=True)
ridge_regressor.fit(X_scaled, Y)
print(ridge_regressor.coef_)

#-> La 1ere colonne ('Acid 1 density') est celle qui a le plus fort poids mais d'autres variables ont aussi un poids 
#   important. L'interpretation est moins nette que dans le cas LASSO. Les erreurs optimales d'estimation sont aussi
#   un peu plus elevees




#QUESTION 3: Afin de comprendre le lien entre l'evolution de la maladie et les variables etudiees,
#            on sélectionne tout au plus 3 variables avec le lasso.
#            -> Utiliser une procedure de type 4-folds pour selectionner typiquement 3 variables
#            -> Les variables selectionnees sont elles stables ?
#            -> Est-ce que le modele garde un bon pouvoir de prediction lorsque avec 3 variables selectionnees


#-> REMARQUE: VU QUE LES DONNEES SONT TRES DIFFICILES, ON UTILISE UN LEAVE-ONE-OUT PLUTOT QU'UN K-FOLD ET 
#             ON ETUDIE LES ERREURS D'APPROXIMATION


lasso_regressor = Lasso(alpha=0.6, fit_intercept=True)  # alpha = 0.6 a tres bien marche pour la question 2
epsilon=0.0001  # variable tres proche de 0 pour tester si un beta_j est nul ou strictement superieur a zero
for train, test in Loo.split(X):
    lasso_regressor.fit(X_scaled[train], Y[train])
  
    y_pred_lasso = lasso_regressor.predict(X_scaled[test])
    normalized_error=(Y[test][0][0]- y_pred_lasso[0])/np.std(Y)   #erreur divisee par l'ecart type des valeurs de Y pour normaliser les valeurs 
    print('variable sortie=',str(test))
    print('-> normalized error=',str(normalized_error))
    print('-> coefs=',str(lasso_regressor.coef_>epsilon))

#-> on peut verifier que les erreurs d'approximation sont honnetes sans plus ce qui s'explique par le fait que la seule relation 
#   viable sur la variable 'Acid 1 density' est assez bruitee
#   La selection de la variable 'Acid 1 density' est cependant tres nette. Le premier coefficient est en effet toujours selectionne
#   est d'autres coefficients sont rarement selectionnes. 

