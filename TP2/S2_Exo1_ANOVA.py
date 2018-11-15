#exemple inspire de  https://www.marsja.se/four-ways-to-conduct-one-way-anovas-using-python/

#1) INIT

#1.1) Importe les modules matplotlib (pour la visualisation de graphes) et pandas (pour la manipulation de donnees)
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

#1.2) charge les donnees
data = pd.read_csv("PlantGrowth.csv")
print(data)
"""
#1.3) represente un resume des données (necessite d'avoir appele matplotlib)
#
#data.boxplot('weight', by='group', figsize=(12, 8))
#plt.show()

#QUESTION 1 : Que signifient les box-plots (ou boite a moustache) ? 
#C'est une representation des donnees classee par groupe avec leur moyenne
# et leurs dispertion
#Vous pouvez taper 'data.boxplot?' pour des informations elementaires

#Les groupes vous semblent-il distribues de maniere similaire ?
#nope
#Les variabilités intra-groupe vous semblent-t-elles similaires ?



#2) STATISTIQUE DE FISHER

 
grps = pd.unique(data.group.values)
d_data = {grp:data['weight'][data.group == grp] for grp in grps}
print(d_data)
 
k = len(grps)  # nombre de groupes
print(k)
N = len(data.values)  # nombre d'observations
print(N)
n = data.groupby('group').size()[0] #nombre d'observations dans chaque groupe (on remarque que tous les groupes on la meme taille)
print(n)
#calcule la variabilite due aux interactions entre groupes (Sum of Squares Between)
print("SSB:")
SSbetween = (sum(data.groupby('group').sum()['weight']**2)/n)- (data['weight'].sum()**2)/N
print(SSbetween)

#calcule la variabilite due aux differences entre les observations (Sum of Squares Within)
sum_y_squared = sum([value**2 for value in data['weight'].values])
SSwithin = sum_y_squared - sum(data.groupby('group').sum()['weight']**2)/n
print('SSW:')
print(SSwithin)

#calcule la variabilite totale (Sum of Squares Total)
SStotal = sum_y_squared - (data['weight'].sum()**2)/N
print('SST:')
print(SStotal)

#defini les degres de liberte
DFbetween = k - 1
DFwithin = N - k
DFtotal = N - 1

#calcule la statistique de Fisher
MSbetween = SSbetween/DFbetween
MSwithin = SSwithin/DFwithin

F = MSbetween/MSwithin


p = stats.f.sf(F, DFbetween, DFwithin)
print(p)
"""
#QUESTION 2 : Faire le lien avec la section 4.3 du cours pour comprendre ce que signifie p.
#probabilie que Ho soit fausse alors qu'on a dit vrai. p-value ?

#QUESTION 3 : Adapter ce code pour etudier la colonne 'f2' des donnees IRIS de 'DataIris.csv'.
#1.2) charge les donnees
data= pd.read_csv("DataIris.csv")
print(data)
 
#1.3) represente un resume des données (necessite d'avoir appele matplotlib)
#

data.boxplot('f2', by='IRIS_type', figsize=(12, 8))
plt.show()

#2) STATISTIQUE DE FISHER
 
grps = pd.unique(data.IRIS_type.values)
d_data = {grp:data['f2'][data.IRIS_type == grp] for grp in grps}
print(d_data)
 
k = len(grps)  # nombre de groupes
print(k)
N = len(data.values)  # nombre d'observations
print(N)
n = data.groupby('IRIS_type').size()[0] #nombre d'observations dans chaque groupe (on remarque que tous les groupes on la meme taille)
print(n)
#calcule la variabilite due aux interactions entre groupes (Sum of Squares Between)
print("SSB:")
SSbetween = (sum(data.groupby('IRIS_type').sum()['f2']**2)/n)- (data['f2'].sum()**2)/N
print(SSbetween)

#calcule la variabilite due aux differences entre les observations (Sum of Squares Within)
sum_y_squared = sum([value**2 for value in data['f2'].values])
SSwithin = sum_y_squared - sum(data.groupby('IRIS_type').sum()['f2']**2)/n
print('SSW:')
print(SSwithin)

#calcule la variabilite totale (Sum of Squares Total)
SStotal = sum_y_squared - (data['f2'].sum()**2)/N
print('SST:')
print(SStotal)

#defini les degres de liberte
DFbetween = k - 1
DFwithin = N - k
DFtotal = N - 1

#calcule la statistique de Fisher
MSbetween = SSbetween/DFbetween
MSwithin = SSwithin/DFwithin

F = MSbetween/MSwithin


p = stats.f.sf(F, DFbetween, DFwithin)
print(p)

#QUESTION 4 : Est-ce qu'un test non-parametrique de kruskal wallis mene aux mêmes conclusions ?
#             Utiliser la fonction scipy.stats.kruskal
from scipy.stats import kruskal
print(kruskal(d_data['Iris-virginica'],d_data['Iris-setosa'],d_data['Iris-versicolor']))