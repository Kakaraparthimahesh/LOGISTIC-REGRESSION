# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 05:45:51 2022

@author: MAHESH
"""
# step1:import the data set

import pandas as pd
df=pd.read_csv("bank-full.csv",sep=";")
df
df.shape
list(df)

# label encoding

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

df.iloc[:,5]

for eachcolumn in range(0,17):
    df.iloc[:,eachcolumn] = LE.fit_transform(df.iloc[:,eachcolumn])

#step2:split the variables as x and y
#split the variables

X=df.iloc[:,0:16]
list(X)
Y=df["y"]

# scatter plot

df.plot.scatter('age','y')
df.plot.scatter('job','y')
df.plot.scatter('marital','y')
df.plot.scatter('education','y')
df.plot.scatter('default','y')
df.plot.scatter('balance','y')
df.plot.scatter('housing','y')
df.plot.scatter('loan','y')
df.plot.scatter('contact','y')            # NO NRRD OF SCATTER PLOT 
df.plot.scatter('day','y')
df.plot.scatter('month','y')
df.plot.scatter('duration','y')
df.plot.scatter('campaign','y')
df.plot.scatter('pdays','y')
df.plot.scatter('previous','y')
df.plot.scatter('poutcome','y')



##step3:model fitting

from sklearn.linear_model import LogisticRegression
logR=LogisticRegression()
logR.fit(X,Y)

#step4:predicting the values

Y_pred=logR.predict(X)
Y
#metrics

from sklearn.metrics import confusion_matrix,accuracy_score

confusion_matrix(Y,Y_pred)

#Step5:calculating the values

acc=accuracy_score(Y,Y_pred)*100
print("Accuracy score is",acc.round(3))


from sklearn.metrics import recall_score,precision_score,f1_score

rscore=recall_score(Y,Y_pred)*100
print("recall_score is",rscore.round(3))


pscore=precision_score(Y,Y_pred)*100
print("precision_score is",pscore.round(3))


f1score=f1_score(Y,Y_pred)*100
print("f1_score is",f1score.round(3))

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#=====================================================================
# Accuracy score         = 88.536
# f1_score               = 28.678
# precision_score        = 52.679
# recall_score           = 19.701

#=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>

# array([[38986,   936],
#       [ 4247,  1042]]

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#=====================================================================



