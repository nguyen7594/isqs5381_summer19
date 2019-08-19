# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 21:07:02 2019

@author: Nguyen7594
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler        
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion    




### IMPORT CSV FILES
### CONTAIN FROM (1982-1983)-(2016-2017), YEAR VAB WILL BE ASSIGNED AS THE LAST YEAR OF THAT SEASON
## YEAR - SEASONS_STATS THE LAST YEAR OF THAT SEASON
## YEAR - ALL_STAR_LIST THE FIRST YEAR OF THAT SEASON
file_1 = 'Seasons_Stats.csv'
file_2 = 'NBA_All_Star_Games.csv'
file_3 = 'NBA_All_Star_Games_addition.csv'
# local file load
file_path_local = 'C:/Users/nguye/Documents/TTU/5381/nba/isqs5381_summer19nn/src/'
#season_stat = pd.read_csv(file_path_local+file_1)
#season_stat.info()
#all_star_list = pd.read_csv(file_path_local+file_2)
#all_star_list.info()
#all_star_list_2 = pd.read_csv(file_path_local+file_3)
#all_star_list_2.info()
# github load
file_path_github = 'https://raw.githubusercontent.com/nguyen7594/isqs5381_summer19nn/master/src/'
def file_import(file_name,FILE_PATH=file_path_github):
    csv_path = os.path.join(FILE_PATH,file_name)
    return pd.read_csv(csv_path)    
season_stat = file_import(file_1)
all_star_list = file_import(file_2)
all_star_list_2 = file_import(file_3)
# first look
#season_stat.info()
#all_star_list.info()
#all_star_list_2.info()


## MERGE 2 ALL_STAR FILES
all_star = pd.concat([all_star_list,all_star_list_2])
#all_star.info()
# Year is changed to the last half of season to match the season_stat
all_star.Year = all_star.Year + 1
# remove 1999 all star game: error - year without all-star
all_star = all_star[all_star['Year'] != 1999]
# 895 vabs
# all_star[all_star['Year'] == 2018].count()
# 2018: 28 obs => rest = 895-28 = 867 obs


### JOIN SEASONS_STATS WITH ALL_STAR FILE FOR BEING ALL_STAR OR NOT
## ALL_STAR SHORT_LIST
all_star['all_star'] = 1
#all_star['all_star'].groupby(all_star['Year']).count()

## REMOVE NA VALUES AND * AT THE END FOR SEASON_STAT PLAYERS NAMES
season_stat.dropna(subset=['Player'],inplace=True)

def remove_(X):
    if X[-1] == '*':
        return X.strip()[:-1]
    else:
        return X.strip()
    
season_stat['Player_'] = season_stat['Player'].map(lambda x: remove_(x))

## REMOVE DUPLICATES: only keep TOT (team ~ 'Tm') stats (Total) record for players - players who were traded in mid-season
season_stat.drop_duplicates(['Year','Player_'],inplace=True)

## Vabs used from season_stat
vabs_selected_ss = ['Year','Player_','Tm','Pos','G','MP','Age','PTS','FG','FG%','2P','2P%','3P','3P%',
                  'FT','FT%','AST','AST%','BLK','BLK%',
                  'DRB','DRB%','ORB','ORB%','STL','STL%',
                  'TOV','TOV%','PF','WS']
#len(vabs_selected_ss)
season_stat = season_stat[vabs_selected_ss]
#season_stat.info()
## Vabs used from all_star
vabs_selected_as = ['Year','Player','Team','all_star']
all_star = all_star[vabs_selected_as] 
#all_star.info()


## JOIN ALL_STAR WITH SEASON_STATS
season_stat_ = pd.merge(season_stat,all_star,left_on=['Year','Player_'],right_on=['Year','Player'],how='left')
#season_stat_.info()
# 853 obs => 14 missing values from all_star

# players would not be matchaed 
#left_out_ = pd.merge(season_stat,all_star,left_on=['Year','Player_'],right_on=['Year','Player'],how='right')
#left_out_.info()
# 14 not matched 
#left_out_[(left_out_['Player_x'].isnull())&(left_out_['Year']!=2018)][['Player_y','Year']]

## FILL MISSING VALUES FOR ALL_STAR vabs: 14 NAs
# Metta World 2004 - 1
season_stat_.loc[(season_stat_['Player_']=='Metta World')&(season_stat_['Year']==2004),'all_star'] = 1 
# Maurice Cheeks 1983,1986,1987,1988 - 4
season_stat_.loc[(season_stat_['Player_']=='Maurice Cheeks')&(season_stat_['Year'].isin([1983,1986,1987,1988])),'all_star'] = 1 
season_stat_.loc[(season_stat_['Player_']=='Maurice Cheeks')&(season_stat_['Year'].isin([1983,1986,1987,1988])),'all_star']
# Micheal Ray 1985 - 1
season_stat_.loc[(season_stat_['Player_']=='Micheal Ray')&(season_stat_['Year']==1985),'all_star'] = 1 
# Joe Barry 1987 - 1
season_stat_.loc[(season_stat_['Player_']=='Joe Barry')&(season_stat_['Year']==1987),'all_star'] = 1 
# Magic Johnson did not play 1992 - 1
# Clifford Robinson 1994 - 1
season_stat_.loc[(season_stat_['Player_']=='Clifford Robinson')&(season_stat_['Year']==1994),'all_star'] = 1 
# Anfernee Hardway 1995,1996,1997,1998 - 4
season_stat_.loc[(season_stat_['Player_']=='Anfernee Hardaway')&(season_stat_['Year'].isin([1995,1996,1997,1998])),'all_star'] = 1 
# Nick Van 1998 - 1
season_stat_.loc[(season_stat_['Player_']=='Nick Van')&(season_stat_['Year']==1998),'all_star'] = 1 
# double check the count: 866 + 1
#season_stat_.info()
# fill others as 0
season_stat_['all_star'].fillna(0,inplace=True)
#season_stat_.info()



### Final csv file output
vabs_selected_output = ['Year','Player_','Tm','Pos','G','MP','Age','PTS','FG','FG%','2P','2P%','3P','3P%',
                  'FT','FT%','AST','AST%','BLK','BLK%',
                  'DRB','DRB%','ORB','ORB%','STL','STL%',
                  'TOV','TOV%','PF','WS','all_star']
season_stat_ = season_stat_[vabs_selected_output]
#season_stat_.to_csv(os.path.join(file_path_local,r'nba_stat_merged.csv'),index=False)
season_stat_.to_csv(os.path.join(file_path_local,r'isqs5381_summer19nn',r'src',r'nba_stat_merged.csv'),index=False)





#### --------------------------------------- ANALYSIS --------------------------------------------####
### FINAL DATA SELECTED
file_merged = 'nba_stat_merged.csv'
nba_stats = file_import(file_merged)
#nba_stats.info()

### DATA UNDERSTANDING ###
nba_8317_all = nba_stats[nba_stats['Year']>1982].copy()
nba_8317_all.to_csv(os.path.join(file_path_local,r'nba_8317_explore.csv'),index=False)
#nba_8317_all.info()
nba_8317_all.hist(figsize=(20,20))
#plt.savefig('hist_project')


### DATA CLEANING ###
## Objective: predict the W/S of the next season
## Lag WS variable back to previous observation
# Index for each players
nba_stats.sort_values(['Player_','Year'],inplace=True)
nba_stats['Index_p'] = nba_stats.groupby(['Player_']).cumcount()
#nba_stats[['Player_','Year','Index_p']]
nba_copy = nba_stats[['Player_','Index_p','WS']].copy()
nba_copy['Index_p'] = nba_copy['Index_p']  - 1
nba_copy.columns = ['Player_2','Index_p2','WS2']
nba_stats = pd.merge(nba_stats,nba_copy,how='left',left_on=['Player_','Index_p'],right_on=['Player_2','Index_p2'])
#nba_stats[['Player_','Index_p','WS','Index_p2','WS2']]
# Drop the last season of each players or only 1-year players
nba_stats_rm = nba_stats.dropna(subset=['WS2']).copy()
# correlation between all_star and WS in the next season
nba_stats_rm['WS2'].corr(nba_stats_rm['all_star'])

#nba_stats_rm.info()
nba_stats_rm.drop(['WS','Player_2','Index_p2','all_star','Index_p'],axis=1,inplace=True)



## Because of limited time for project and many missing values from previous periods
## We only focus on analyzing the data from 1983-2016
## Thus, use individual stats from '83-'16 to predict WS in '84-'17
nba_8316 =  nba_stats_rm[nba_stats_rm['Year']>1982].copy()
#nba_stats_rm[nba_stats_rm['Year']>1986].info()
#nba_8316.info()
#nba_8316.describe()
#nba_8316.hist()
# most total indicators e.g. pts, 2p, 3p, ... are right-skewed  
# thus it is better to average these total indicators by Games played in given season  

### EXPORT DATASETS TO DEVELOP MODELS IN R ###
# All set    
nba_8316.to_csv(os.path.join(file_path_local,r'nba_8316.csv'),index=False)



### MODEL DEVELOPMENT ### --- IN PROGRESS ---

## Split training-test sets as ratio of 80-20, stratied by Year 
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=25)  
for train_index, test_index in split.split(nba_8316,nba_8316['Year']):
    strat_train_set = nba_8316.iloc[train_index]
    strat_test_set = nba_8316.iloc[test_index]
#nba_8716.info()    
#strat_train_set.info()  
#strat_train_set.head()  
#strat_test_set.info()  
#strat_test_set.head()  
#nba_8716['Year'].value_counts()/len(nba_8716)
#strat_train_set['Year'].value_counts()/len(strat_train_set)
#strat_test_set['Year'].value_counts()/len(strat_test_set)

# Split training-validation sets as ratio 80-10, stratied by Year     
split_train = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=25)  
for ttrain_index, tvalid_index in split_train.split(strat_train_set,strat_train_set['Year']):
    ttrain_set = strat_train_set.iloc[ttrain_index]
    tvalid_set = strat_train_set.iloc[tvalid_index]
## train: ttrain_set, valid: tvalid_set, test: strat_test_set
    
#### DATA VISUALIZATION ####
cor_matrix = strat_train_set.corr()
# WS2 correlation
cor_matrix['WS2'].plot.barh(figsize=(10,8))
#plt.savefig('WS2_cor')
# all correlations
plt.figure(figsize=(15,15))
sns.heatmap(cor_matrix)
#plt.savefig('all_cor')
# remove FG, keep PTS only
# convert total indicators and Minitues to average per G
# remove G 
# remove ORB%, BLK%, TOV, 3P%, Age, Year

    
#### DATA CLEANING ####
## Feature selection ##
num_feature_names = ["G","MP","PTS","FG%",
                     "2P","2P%","3P","FT",
                     "FT%","AST","AST%","BLK",
                     "DRB","DRB%","ORB","TOV%",
                     "STL","STL%","PF"]
cat_feature_names = ['Pos']


class xdfselector(BaseEstimator,TransformerMixin):
    def __init__(self,feature_names):
        self._feature_names = feature_names
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        return X[self._feature_names]

## Average some variables:
ave_feature = ["MP","PTS","2P",
              "3P","FT","AST",
              "BLK","DRB","ORB",
              "STL","PF"]
        
class ave_xvab(BaseEstimator,TransformerMixin):
    def __init__(self,ave_feature):
        self._ave_feature = ave_feature
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        X.loc[:,self._ave_feature] = X.loc[:,self._ave_feature]/X.loc[:,"G"]
        return X
    
## Create Dummy variables for Position - Pos
#class dummy_pos(BaseEstimator,TransformerMixin):
#    def __init__(self):
#        self._pos = ['PG','SG','SF','PF','C']
#    def fit(self,X,y=None):
#        return self
#    def transform(self,X,y=None):
#        for i in self._pos:
#            X.assign(i=0)
#        for i, pos in enumerate(X.Pos):
#            indices = X.columns.get_indexer(pos.split('-'))
#            X.iloc[i,indices] = 1
#        return X   
#from sklearn.preprocessing import MultiLabelBinarizer
#multi = MultiLabelBinarizer(classes=['PG','SG','SF','PF','C'])
#multi.fit_transform(ttrain_set['Pos'])
#ttrain_set['Pos'].head()
 
       
## Pipeline ##
num_pipeline = Pipeline([('features_select',xdfselector(num_feature_names)),
                         ('average_values',ave_xvab(ave_feature)),
                         ('missing_values',SimpleImputer(strategy='constant',fill_value=0)),
                         ('feature_range',MinMaxScaler())])    
num_df = num_pipeline.fit_transform(ttrain_set) 
num_df[:,4]
    

cat_pipeline = Pipeline([('features_select',xdfselector(cat_feature_names)),
                         ('dummy_variables',MultiLabelBinarizer()),
                         ('missing_values',SimpleImputer(strategy='constant',fill_value=0))])    

    
    
cat_pipeline.fit_transform(ttrain_set)
preprocess_pipeline=FeatureUnion(transformer_list=[
        ('num_pipeline',num_pipeline),
        ('cat_pipeline',cat_pipeline)])
    
x_train = preprocess_pipeline.fit_transform(ttrain_set)   






 
 

 

 


 

 

 
 
 

 
 

 







