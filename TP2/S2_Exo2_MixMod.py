
#Exercice inspire de http://www.statsmodels.org/stable/mixed_linear.html

import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pandas as pd


data = sm.datasets.get_rdataset("dietox", "geepack").data
grps = pd.unique(data.Pig.values)
d_data = {grp:[data['Weight'][data.Pig == grp],
			   data['Time'][data.Pig == grp]] for grp in grps}
#print(d_data)
pig_select=[8439,4756,4643]
for pig,w in d_data.items():
	weight=w[0]
	time=w[1]
	plt.plot(time,weight)
#plt.show()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#QUESTION 1 : En s'inspirant de ce qui s'est fait dans l'exercice 
#precedent, représenter les courbes d'evolution du poids dans le temps 
#pour les cochons 8439, 4756, 4643.
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

md = smf.mixedlm("Weight ~ Cu", data, groups=data["Pig"])
print(md)
mdf = md.fit()
print(mdf.summary())


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#QUESTION 2 : Taper 'mdf?' ou mdf.[TAB] pour comprendre ce que contient 
#             mdf
#
#QUESTIONS 3 : Quel est le poids moyen a l'orignie et la croissance 
#              moyenne (au sens des moindres carres) ?
#              Quels sont les cochons les plus lourds et les plus leger ?
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#QUESTION 4 : Est-ce que le groupe 'Cu' semble avoir une influence sur 
#             le poids ?
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#non