#!/usr/bin/env python
# coding: utf-8

# #    Structuration d'un projet d’apprentissage machine supervisé                     Projet: Prédire le prix d'une maison.
# 
# Objectifs du projet:
# Créer un model de ML pour prédire le prix d’une maison  en fonctions des données qui décrivent les caractéristiques et le prix certaines maisons.

# Etape 1: Chargement et Inspection des données.

# In[76]:


# importation des bibliothèques.

# Chargement et analyse des données.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Evaluation du model
from sklearn.model_selection import (train_test_split, 
                                     cross_val_score,
                                     learning_curve,
                                     GridSearchCV, 
                                     KFold)
# pre-processing
from sklearn.preprocessing import (StandardScaler,
                                   LabelEncoder, 
                                   OneHotEncoder)
# Encodage
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder,StandardScaler,MinMaxScaler
from sklearn.compose import make_column_selector,make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVC


# In[60]:


#Chargement des données.
data = pd.read_csv('C:\\Users\\PC\\Downloads\\house-prices-advanced-regression-techniques\\train.csv')
#Inspection des données.
#Inspection du debut
data.head()


# In[61]:


#description de la data
data.describe()


# In[62]:


#information sur les types
data.info()


# Type des données.
# Label : SalePrice.
# Types de variables : 43 variables qualitatives et 38 variables quantitatives.

# In[63]:


# Forme des données
data.shape


# Etape 2: Traitement des valeurs manquantes.

# In[64]:


# Définition d'une fonction pour ressortir des statistiques sur les valeurs manquantes.
def summarize_missingness(data):
    nulls = data.isnull()
    counts = nulls.sum()
    percs = nulls.mean().mul(100.)
    
    nulls_data = pd.DataFrame({'Count of missing values': counts, 'Percentage of missing values': percs}, 
                            index=counts.index)
    
    display(nulls_data)
    
vars_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]

for dataframe in [data]:
    summarize_missingness(dataframe[vars_with_na])


# Nous observons que:
# 19 variables ont des valeurs manquantes avec 03 variables (PoolQC, MiscFeature, Alley) avec un taux de valeurs manquantes > 90%.
# 03 groupes de variables enregistrent le même nombres de variables. Celles-ci sont probablement liées, il s'agit des variables :
#     * BsmtQual, BsmtCond, BsmtFinType1 (37%).
#     *BsmtFinType2, BsmtExposure (37%).
#     *GarageType, GarageYrBlt, GarageFinish, GarageQual, GarageCond (81).

# In[65]:


# Suppression des colonnes qui enregistrent plus de 90/100 de valeurs manquantes.

to_drop = []
for var in vars_with_na:
    if data[var].isnull().mean() > 0.9:
        to_drop.append(var)
        
data.drop(columns=to_drop, inplace=True)
vars_with_na = [var_with_na for var_with_na in vars_with_na if var_with_na not in to_drop]
summarize_missingness(data[vars_with_na])


# In[66]:


data.shape


# pre-processing

# In[67]:


# Suppression de la colonne ID.
data = data.drop('Id', axis = 1)


# In[68]:


data.shape


# In[70]:


#Définition du Label(y) et des Features(X) 
X = data.drop('SalePrice',axis=1)
y = data['SalePrice']


# In[72]:


# Séparation de la data et jeux d'entrainement et de test
SEED = 123
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    data['SalePrice'], 
                                                    test_size=0.15, random_state=SEED)


# 

# In[73]:


X_train.shape


# In[74]:


X_test.shape


# In[77]:


#Division des variables en fonction des catégories.
cat_var= make_column_selector(dtype_exclude=np.number)
num_var = make_column_selector(dtype_include=np.number)


# In[78]:


#Création des pipelines.
pipelinec = make_pipeline(SimpleImputer(strategy= 'most_frequent'),OneHotEncoder( handle_unknown='ignore'))
pipelinen = make_pipeline(SimpleImputer(),MinMaxScaler())


# In[79]:


Prp = make_column_transformer((pipelinec,cat_var),
               (pipelinen,num_var))


# In[80]:


KNeighbors = make_pipeline(Prp,KNeighborsClassifier())
linearRe = make_pipeline(Prp,LinearRegression())
rand_class = make_pipeline(Prp,RandomForestRegressor(random_state=42))


# In[83]:


models=[KNeighbors,linearRe,rand_class]
for i in models :
        i.fit(X_train,y_train)
        print(f'{i.score(X_train,y_train)}')


# Le meilleur score est celui de LinearRegression.

# In[92]:


#Prédiction.
linearRe.predict(X_test).round(2)


# In[ ]:




