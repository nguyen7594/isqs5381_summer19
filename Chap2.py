# -*- coding: utf-8 -*-
"""
Created on Wed May 15 09:08:09 2019

@author: Nguyen7594
"""


## PRACTICE EXERCISE FROM HANDS-ON ML ##
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tarfile
from six.moves import urllib
# Split the test-train datasets
from sklearn.model_selection import StratifiedShuffleSplit   
# Fill missing value
from sklearn.impute import SimpleImputer
# Encode category variables
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
# Custom Transformers
from sklearn.base import BaseEstimator, TransformerMixin
# Pipeline for transformation for numerical variables
from sklearn.pipeline import Pipeline
# Scale the data
from sklearn.preprocessing import StandardScaler
# Full pipeline for transformation
from sklearn.compose import ColumnTransformer
# Linear regression
from sklearn.linear_model import LinearRegression 
# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
# Mean Squared Error
from sklearn.metrics import mean_squared_error
# Mean Absolute Error
from sklearn.metrics import mean_absolute_error
# Cross Validation
from sklearn.model_selection import cross_val_score
# Fine tune model: GridSearch
from sklearn.model_selection import GridSearchCV 

pd.options.display.max_rows = 20
##------------------ Chap 1 -----------------------------##
# Load the data
oecd_bli = pd.read_csv("C:/Users/nguye/Documents/Python/Hands-on ML/oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("C:/Users/nguye/Documents/Python/Hands-on ML/gdp_per_capita.csv",thousands=',',
                             delimiter='\t',encoding='latin1', na_values="n/a")

oecd_bli.info()
gdp_per_capita.info()


##------------------ Chap 2 -----------------------------##

## GET THE DATA
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
# open rar file from url
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
fetch_housing_data()    
# import csv file from rar file    
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
housing = load_housing_data()
# take a quick look at data structure
housing.info()
housing.ocean_proximity.value_counts()
housing.describe()
# histogram
housing.hist(bins=50, figsize=(20,15))
plt.show()
# Create the Test set
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)  #From sklearn.model_selection
for train_index,test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
#double-check
housing["income_cat"].value_counts()/len(housing)    
strat_train_set["income_cat"].value_counts()/len(strat_train_set)    
strat_test_set["income_cat"].value_counts()/len(strat_test_set)    
# Drop income_cat in both test and train sets
for set_ in [strat_train_set,strat_test_set]:
    set_.drop("income_cat",inplace=True,axis=1)
#strat_train_set.info()
#strat_test_set.info()
housing = strat_train_set.copy()


## Discovery and Visualization
housing.plot(kind='scatter',x='longitude',y='latitude',alpha=0.1)

housing.plot(kind='scatter',x='longitude',y='latitude',alpha=0.4,s=housing['population']/100,label='population',figsize=(10,7),
            c='median_house_value',cmap=plt.get_cmap('jet'),colorbar=True)
plt.legend()
#correlation matrix
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)
attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]
pd.plotting.scatter_matrix(housing[attributes], figsize=(6,4))
housing.plot(kind="scatter", x="median_income", y="median_house_value",alpha=0.1)
#additional variables
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"] 
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

## Prepare data for ML - Data cleaning
# Separate predictors and labels
housing = strat_train_set.drop("median_house_value", axis=1) #predicctors
housing_labels = strat_train_set["median_house_value"].copy() #labels - response
# housing numerical variables only
housing_num = housing.drop("ocean_proximity", axis=1)
# Missing value
imputer = SimpleImputer(strategy='median') # from sklearn.imputer
X = imputer.fit_transform(housing_num)
housing_tr = pd.DataFrame(X,columns=housing_num.columns)
housing_tr.info()
# Encode category variables
# OrdinalEncoder
housing_cat = strat_train_set[['ocean_proximity']]
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]
ordinal_encoder.categories_
# OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded)
housing_cat_1hot.toarray()
# Custom Transformer
rooms_ix, bedrooms_ix, population_ix, household_ix = 3,4,5,6
class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
    def __init__(self,add_bedrooms_per_room=True): # no *args or ** kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        rooms_per_household = X[:,rooms_ix] / X[:,household_ix]
        population_per_household = X[:,population_ix] / X[:,household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:,bedrooms_ix] / X[:,rooms_ix]
            return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]
        else:
            return np.c_[X,rooms_per_household,population_per_household]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.fit_transform(housing.values) 
# Pipeline for numerical variables
num_pipeline = Pipeline([
        ('imputer',SimpleImputer(strategy='median')),
        ('attribs_adder',CombinedAttributesAdder()),
        ('std_scaler',StandardScaler())])
housing_num_tr = num_pipeline.fit_transform(housing_num)
# Pipeline for all variables
#housing.info()
num_attribs = list(housing_num)
num_attribs
cat_attribs = ['ocean_proximity']
full_pipeline = ColumnTransformer([
        ('num',num_pipeline,num_attribs),
        ('cat',OneHotEncoder(),cat_attribs)])
housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared


## Model selection 
# Linear regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels,housing_predictions)
np.sqrt(lin_mse)
# Decision Tree Regressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared,housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels,housing_predictions)
np.sqrt(lin_mse)
# Ensemble learning - Random Forest Regressor
forest_reg = RandomForestRegressor()



# Cross-Validation
scores = cross_val_score(tree_reg,housing_prepared,housing_labels,
                         scoring='neg_mean_squared_error',cv=10)
rmse_scores = np.sqrt(-scores)
rmse_scores.mean()
rmse_scores.std()




## Fine tune model
param_grid =[{'n_estimators':[3,10,30],'max_features':[2,4,6,8]},
              {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]}]
grid_search = GridSearchCV(forest_reg,param_grid,cv=5,
                           scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared,housing_labels)
grid_search.best_params_
grid_search.best_estimator_




## Analyze the best models and its errors
feature_importances = grid_search.best_estimator_.feature_importances_
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)



## Evaluate on test set
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse) 
