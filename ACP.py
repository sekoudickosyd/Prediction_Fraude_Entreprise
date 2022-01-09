import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def acp_cercle_correlation(df):
    
    X = df.copy()
    
    n = X.shape[0]
    p = X.shape[1]
    
    sc = StandardScaler()
    
    Z = sc.fit_transform(X)
    
    acp = PCA(svd_solver='full')
    coord = acp.fit_transform(Z)
    
    eigval = (n-1)/n*acp.explained_variance_
    
    sqrt_eigval = np.sqrt(eigval)

    corvar = np.zeros((p,p)) 
    
    for k in range(p): 
        
        corvar[:,k] = acp.components_[k,:] * sqrt_eigval[k]
    
    
    #cercle des corrélations 
    fig, axes = plt.subplots(figsize=(8,8)) 
    axes.set_xlim(-1,1) 
    axes.set_ylim(-1,1) # affichage des étiquettes (noms des variables) 
    for j in range(p): 
        plt.annotate(X.columns[j],(corvar[j,0],corvar[j,1])) # ajouter les axes
        plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1) 
        plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)

    #ajouter un cercle 
    cercle = plt.Circle((0,0),1,color='blue',fill=False) 
    axes.add_artist(cercle)
    plt.show()
    return "Cercle des correlations"


def acp_representation_individus(df):
    
    X = df.copy()
    
    n = X.shape[0]
    p = X.shape[1]
    
    sc = StandardScaler()
    
    Z = sc.fit_transform(X)
    
    acp = PCA(svd_solver='full')
    coord = acp.fit_transform(Z)
    
    fig, axes = plt.subplots(figsize=(8,8)) 

    axes.set_xlim(-6,6) #même limites en abscisse

    axes.set_ylim(-6,6) #et en ordonnée #placement des étiquettes des observations
    
    plt.title("Representation des individus dans le premier plan factoriel")

    plt.scatter(coord[:,0],coord[:,1], c=X.iloc[:,-1]) 
    
    # ajouter les axes
    
    plt.plot([-6,6],[0,0],color='silver',linestyle='-',linewidth=1) 
    
    plt.plot([0,0],[-6,6],color='silver',linestyle='-',linewidth=1)
    
    return "Individus dans le premier plan factoriel"
    
    

