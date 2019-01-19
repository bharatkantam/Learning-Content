#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 10:22:56 2019

@author: A.V.A.Bharat Kumar
"""

# Reading an excel file using Python 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# To open Workbook 
data = pd.read_excel('Ass-1.xlsx')
data1 = data

#renaming the column names
data.columns = ['Labels', 'A','B','C','D','E','F','G','H','I','J']

Cat_1 = np.array(data[data.Labels == 1])
Cat_2 = np.array(data[data.Labels == 2])

#scatter plotting
def plot_Cat(A_1, B_2):
    for i in range(1,10):
        for j in range (i+1, 11):  
            plt.scatter(Cat_1[:,i],Cat_1[:,j])
            plt.scatter(Cat_2[:,i],Cat_2[:,j])
            plt.xlabel(i)
            plt.ylabel(j)
            plt.title('Scatter plot for Binary Classification')
            plt.show()

plot_Cat(Cat_1, Cat_2)

# OR the other process
### plotting scatter 

data.columns = ['Labels', 'A','B','C','D','E','F','G','H','I','J']
A = plt.scatter(data.Labels,data.A)
A = plt.xlabel('Labels')
A = plt.ylabel('A')
A = plt.title('Labels Vs A')

B = plt.scatter(data.Labels,data.B)
B = plt.xlabel('Labels')
B = plt.ylabel('B')
B = plt.title('Labels Vs B')

C = plt.scatter(data.Labels,data.C)
C = plt.xlabel('Labels')
C = plt.ylabel('C')
C = plt.title('Labels Vs C')

D = plt.scatter(data.Labels,data.D)
D = plt.xlabel('Labels')
D = plt.ylabel('D')
D = plt.title('Labels Vs D')

E = plt.scatter(data.Labels,data.E)
E = plt.xlabel('Labels')
E = plt.ylabel('E')
E = plt.title('Labels Vs E')

F = plt.scatter(data.Labels,data.F)
F = plt.xlabel('Labels')
F = plt.ylabel('F')
F = plt.title('Labels Vs F')

G = plt.scatter(data.Labels,data.G)
G = plt.xlabel('Labels')
G = plt.ylabel('G')
G = plt.title('Labels Vs G')

H = plt.scatter(data.Labels,data.H)
H = plt.xlabel('Labels')
H = plt.ylabel('H')
H = plt.title('Labels Vs H')

I = plt.scatter(data.Labels,data.I)
I = plt.xlabel('Labels')
I = plt.ylabel('I')
I = plt.title('Labels Vs I')

J = plt.scatter(data.Labels,data.J)
J = plt.xlabel('Labels')
J = plt.ylabel('J')
J = plt.title('Labels Vs J')

# PCA analysis

data = np.array(data)
x = data[:,1:11]
y = data[:,0]
# Standardizing the features
x = StandardScaler().fit_transform(x.T)
print(x)

from sklearn.decomposition import PCA

pca = PCA()
pca.fit_transform(x)


Z = np.round(pca.explained_variance_ratio_*100, decimals=1)

labels1 = ['PC' + str(x) for x in range(1,len(Z)+1)]
plt.bar(x = range(1,len(Z)+1), height = Z, tick_label = labels1)


from sklearn import preprocessing

correlation = data1.corr()['Labels'][data1.corr()['Labels'] < 1].abs()

correlation.head(10)