
#Exercice inspire de http://www.statsmodels.org/stable/mixed_linear.html

import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


data = sm.datasets.get_rdataset("dietox", "geepack").data

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#QUESTION 1 : En s'inspirant de ce qui s'est fait dans l'exercice 
#precedent, représenter les courbes d'evolution du poids dans le temps 
#pour les cochons 8439, 4756, 4643.
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

plt.plot(data['Time'][data.Pig == 8442],data['Weight'][data.Pig == 8442],'r')
plt.plot(data['Time'][data.Pig == 4756],data['Weight'][data.Pig == 4756],'g')
plt.plot(data['Time'][data.Pig == 4643],data['Weight'][data.Pig == 4643],'b')
plt.show()


md = smf.mixedlm("Weight ~ Time", data, groups=data["Pig"])
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

-> mdf.fe_params.Intercept
-> mdf.fe_params.Time
-> mdf.random_effects -> effets aléatoires (ecart a la moyenne)

highest=-1.
lowest=100.
for key in mdf.random_effects.keys():
  if (highest<mdf.random_effects[key].Group):
    highest=mdf.random_effects[key].Group
    highest_Key=key
  if (lowest>mdf.random_effects[key].Group):
    lowest=mdf.random_effects[key].Group
    lowest_Key=key

plt.plot(data['Time'][data.Pig == highest_Key],data['Weight'][data.Pig == highest_Key],'r')
plt.plot(data['Time'][data.Pig == 8442],data['Weight'][data.Pig == 8442],'g')
plt.plot(data['Time'][data.Pig == 4756],data['Weight'][data.Pig == 4756],'g')
plt.plot(data['Time'][data.Pig == 4643],data['Weight'][data.Pig == 4643],'g')
plt.plot(data['Time'][data.Pig == lowest_Key],data['Weight'][data.Pig == lowest_Key],'b')
plt.show()


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#QUESTION 4 : Est-ce que le groupe 'Cu' semble avoir une influence sur 
#             le poids ?
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

md2 = smf.mixedlm("Weight ~ Time", data, groups=data["Cu"])
mdf2 = md2.fit()

mdf.fe_params
mdf2.fe_params

mdf2.random_effects
