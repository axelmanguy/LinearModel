import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

#1) LECTURE, EXTRACTION ET VISUALISATION DES DONNEES

dataframe=pandas.read_csv("./MedicalData1.csv",sep=';',decimal=b',')

listColNames=list(dataframe.columns)


XY=dataframe.values
ColNb_Y=listColNames.index('Disease progression')


Y=XY[:,ColNb_Y].reshape((XY.shape[0],1))   #reshape is to make sure that Y is a column vector
X = np.delete(XY, ColNb_Y, 1)

X_scaled = preprocessing.scale(X)

listColNames.pop(ColNb_Y)     #to make it contains the column names of X only


for Col in range(len(listColNames)):
  plt.plot(X[:,Col],Y[:],'.')
  plt.xlabel(listColNames[Col])
  plt.ylabel('Disease progression')
  plt.show()



#QUESTION 1 : Observez les donnees unes par unes. Est-ce que vous identifiez visuellement des liens entre 
#certaines variables et la variable 'Disease progression'. Si oui, lesquels ?


#QUESTION 2 :   On se demande si il est possible de predire le niveau de 'Disease progression' à partir de
#               de la variable 'Acid 1 density'. 
#QUESTION 2.1 : Effectuez une regression lineaire simple entre ces deux variables et predisez
#               ensuite les valeurs de 'Disease progression' a l'aide de 'Acid 1 density'. Vous pourrez 
#               evaluer la qualité des predictions a l'aide du coefficient de determination R2.
#QUESTION 2.2 : Evaluez la stabilite des predictions a l'aide d'une methode de validation croisee de type
#               4-folds.
#QUESTION 2.3 : Auriez-vous eu de meilleurs resultats en predisant 'Disease progression' a l'aide de la
#               variable 'Biomarker 8' ou 'Pressure 1'?
#QUESTION 2.4 : Peut-on enfin dire si on observe une relation significative entre 'Disease progression'
#               et (independament) 'Acid 1 density' ou 'Biomarker 8' ou 'Pressure 1'. On peut le valider
#               en testant si les coefficients des pentes obtenues entre 'Disease progression' et chacune
#               de ses trois variables par regression lineaire simples sont significativement differentes
#               de 0.

#QUESTION 3 :   On s'interesse maintenant au lien entre la variable 'Disease progression' et 'Biomarker 5'.
#               On peut remarquer que ces donnees contiennent deux valeurs aberrantes.
#QUESTION 3.1 : Quelles sont les valeurs aberrantes et quel va etre leur impact lors de l'estimation
#               d'une relation lineaire entre ces deux variables ?
#QUESTION 3.2 : Definissez une procedure de detection automatique de ces variables basee sur la distance de 
#               Cook.
#QUESTION 3.3 : Pourriez vous plus simplement detecter ces observations aberrantes a l'aide des residus ?
#               Si oui, decrivez comment, sans necessairement coder la solution.
#QUESTION 3.4 : Comment traiteriez-vous les donnees aberrantes dans le contexte de la regression lineaire 
#               entre 'Disease progression' et 'Biomarker 5' pour estimer un lien pertinent entre ces
#               deux variables

#QUESTION 4 :   Une fois les observations aberrantes de 'Biomarker 5' traitees, on souhaite selectionner les
#               variables de 'X' qui permettent de prédire au mieux 'Disease progression' a l'aide de la 
#               regression multiple regularisee.
#QUESTION 4.1 : Expliquez pourquoi vous effecturez cette procedure sur 'X_scaled' plutot que 'X' ?
#QUESTION 4.2 : Expliquez pourquoi vous regulariserez les coefficients estimes avec une regularisation de
#               type LASSO plutot que RIDGE ?
#QUESTION 4.3 : Codez la procedure de selection des variables optimales en parametrant a la main le poids
#               entre la qualite de prediction et le niveau de regularisation
#QUESTION 4.4 : Codez la procedure automatique de parametrisation de ce poids, de sorte a ce q'un maximum 
#               de trois variables soit typiquement selectionne et que la qualite de prediction soit optimale.
#               Une procédure de validation croisee de type leave-one-out sera idealement utilisee.
#               La selection des variables est-elle stable ?


#QUESTION 5 :   On s'interesse enfin au fichier 'MedicalData2.csv' et non 'MedicalData1.csv' qui contient une
#               colonne qualitative supplementaire 'Pathology type'.
#QUESTION 5.1 : Quelles sont les differences entre les structures du fichier 'MedicalData2.csv' et du fichier
#               'MedicalData1.csv' en plus de cette colonne supplementaire
#QUESTION 5.2 : Est-ce qu'une variable semble bien expliquer la classe de 'Pathology type'. Quelle demarche
#               vous semble la plus pertinente pour identifer cette variable ?
 


