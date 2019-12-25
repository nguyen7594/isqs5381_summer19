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
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MinMaxScaler        
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor


### IMPORT CSV FILES
### CONTAIN FROM (1982-1983)-(2016-2017), YEAR VAB WILL BE ASSIGNED AS THE LAST YEAR OF THAT SEASON
## YEAR - SEASONS_STATS THE LAST YEAR OF THAT SEASON
## YEAR - ALL_STAR_LIST THE FIRST YEAR OF THAT SEASON
file_1 = 'Seasons_Stats.csv'
file_2 = 'NBA_All_Star_Games.csv'
file_3 = 'NBA_All_Star_Games_addition.csv'

# Github load
file_path_github = 'https://raw.githubusercontent.com/nguyen7594/isqs5381_summer19nn/master/src/'
def file_import(file_name,FILE_PATH=file_path_github):
    csv_path = os.path.join(FILE_PATH,file_name)
    return pd.read_csv(csv_path)    
season_stat = file_import(file_1)
all_star_list = file_import(file_2)
all_star_list_2 = file_import(file_3)

# First Look
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
## 3 datasets: train: ttrain_set, valid: tvalid_set, test: strat_test_set
    
    
    
    
#### DATA VISUALIZATION ####
# Missing values
na_cnt = strat_train_set.isnull().sum()    
na_cnt.plot.barh(figsize=(10,8))
# Correlation matrix
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
class ave_xvab(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        ave_MP = X.loc[:,"MP"]/X.loc[:,"G"]
        ave_PTS = X.loc[:,"PTS"]/X.loc[:,"G"]
        ave_2P = X.loc[:,"2P"]/X.loc[:,"G"]
        ave_3P = X.loc[:,"3P"]/X.loc[:,"G"]
        ave_FT = X.loc[:,"FT"]/X.loc[:,"G"]
        ave_AST = X.loc[:,"AST"]/X.loc[:,"G"]
        ave_BLK = X.loc[:,"BLK"]/X.loc[:,"G"]
        ave_DRB = X.loc[:,"DRB"]/X.loc[:,"G"]
        ave_ORB = X.loc[:,"ORB"]/X.loc[:,"G"]
        ave_STL = X.loc[:,"STL"]/X.loc[:,"G"]
        ave_PF = X.loc[:,"PF"]/X.loc[:,"G"]
        return pd.concat([X,pd.DataFrame({'ave_MP':ave_MP,'ave_PTS':ave_PTS,'ave_2P':ave_2P,
                                          'ave_3P':ave_3P,'ave_FT':ave_FT,'ave_AST':ave_AST,
                                          'ave_BLK':ave_BLK,'ave_DRB':ave_DRB,'ave_ORB':ave_ORB,
                                          'ave_STL':ave_STL,'ave_PF':ave_PF})],axis=1)
# Final numerical variables used for model development        
final_xvab = ["ave_MP","ave_PTS","FG%",
              "ave_2P","2P%","ave_3P","ave_FT",
              "FT%","ave_AST","AST%","ave_BLK",
              "ave_DRB","DRB%","ave_ORB","TOV%",
              "ave_STL","STL%","ave_PF"]

## Pipeline ##
num_pipeline = Pipeline([('features_select',xdfselector(num_feature_names)),
                         ('average_values',ave_xvab()),
                         ('final_features_select',xdfselector(final_xvab)),
                         ('missing_values',SimpleImputer(strategy='constant',fill_value=0)),
                         ('feature_range',MinMaxScaler())])
num_df = num_pipeline.fit_transform(ttrain_set) 
## Total number predictor variables = 18
## Label
ws_label = ttrain_set['WS2'].copy()




#### MODEL SELECTION ####
## Linear Regression (plain)
lin_reg = LinearRegression()
lin_reg_score = cross_val_score(lin_reg,num_df,ws_label,scoring='neg_mean_squared_error',cv=10)
lin_scores = np.sqrt(-lin_reg_score)
lin_scores.mean() #2.1880996328813636
lin_scores.std()  #0.07587824076684392 

# Ridge Regression 

# Lasso Regression

# Elastic Net 



## Stochastic Gradient Descent (None Regularization)
sgd_reg = SGDRegressor(n_iter=50, penalty=None, eta0=0.1,random_state=25)
sgd_reg_score = cross_val_score(sgd_reg,num_df,ws_label,scoring='neg_mean_squared_error',cv=10)
sgd_scores = np.sqrt(-sgd_reg_score)
sgd_scores.mean() #2.192746433953347
sgd_scores.std()  #0.07518484726380842




#### ------------ CONFERENCE PAPER ------------ #### 
### OBJECTIVE: PREDICT NBA ALL STARS BASED ON PREVIOUS SEASON STATS
### DATA IMPORT
file_1 = 'Seasons_Stats.csv'
# All Star Roster
file_4 = 'NBA_All_Star_Games_195051to201718.csv'
# All NBA Lists
file_5 = 'all_nba_lists.csv'
# Local file load
file_path_local = 'C:/Users/nguye/Documents/TTU/5381/nba/isqs5381_summer19nn/src/'
# Players stat
season_stat = pd.read_csv(file_path_local+file_1)
season_stat.info()
# All Star rosters
all_star_list = pd.read_csv(file_path_local+file_4)
all_star_list.info()
# All NBA lists
all_nba_list = pd.read_csv(file_path_local+file_5)
all_nba_list.info()
 
### DATA CLEANING --------------------------------------------------------------------
## DATA MERGING
## ALL_STAR SHORT_LIST
all_star_list['all_star'] = 1
 
## REMOVE NA VALUES AND * AT THE END FOR SEASON_STAT PLAYERS NAMES
season_stat.dropna(subset=['Player'],inplace=True)

def remove_(X):
    if X[-1] == '*':
        return X.strip()[:-1]
    else:
        return X.strip()
    
season_stat['Player_'] = season_stat['Player'].map(lambda x: remove_(x))

## REMOVE DUPLICATES: only keep TOT (team ~ 'Tm') stats (Total) record for players 
## specifically for players who were traded in mid-season
season_stat.drop_duplicates(['Year','Player_'],inplace=True) 
 
## Vabs used from season_stat
vabs_selected_ss = ['Year','Player_','Pos','G','MP','Age','PTS','FG','FG%','2P','2P%','3P','3P%',
                  'FT','FT%','AST','AST%','BLK','BLK%',
                  'DRB','DRB%','ORB','ORB%','STL','STL%',
                  'TOV','TOV%','PF']
#len(vabs_selected_ss)
season_stat = season_stat[vabs_selected_ss]
#season_stat.info()

## Vabs used from all_star
vabs_selected_as = ['Year','Player','Team','all_star']
all_star_list = all_star_list[vabs_selected_as] 

 
## JOIN ALL_STAR WITH SEASON_STATS
## FOR SEASONS (1982-1983)-(2016-2017)
# There will be a time lapse. All Star will show if he was chosen in all star roster for next season
# remove 1999 all star game: error - year without all-star
all_star_list = all_star_list[all_star_list['Year'] != 1999]

# Merge season_stat and all_star_list
player_info = pd.merge(season_stat,all_star_list,left_on=['Year','Player_'],right_on=['Year','Player']
                        ,how='left')
player_info = player_info[player_info['Year'] > 1982]
player_info.info()
all_star_list[all_star_list['Year'] > 1982].info()
all_star_list[all_star_list['Year'] > 2017].info()
896-854-28
# 14 missing values
# After filling missing values
896-867-28-1
# No missing value 


## FILL MISSING VALUES FOR ALL_STAR vabs: 14 NAs
# Metta World 2004 - 1
player_info.loc[(player_info['Player_']=='Metta World')&(player_info['Year']==2004),'all_star'] = 1 
# Maurice Cheeks 1983,1986,1987,1988 - 4
player_info.loc[(player_info['Player_']=='Maurice Cheeks')&(player_info['Year'].isin([1983,1986,1987,1988])),'all_star'] = 1 
player_info.loc[(player_info['Player_']=='Maurice Cheeks')&(player_info['Year'].isin([1983,1986,1987,1988])),'all_star']
# Micheal Ray 1985 - 1
player_info.loc[(player_info['Player_']=='Micheal Ray')&(player_info['Year']==1985),'all_star'] = 1 
# Joe Barry 1987 - 1
player_info.loc[(player_info['Player_']=='Joe Barry')&(player_info['Year']==1987),'all_star'] = 1 
# Magic Johnson did not play 1992 - 1
# Clifford Robinson 1994 - 1
player_info.loc[(player_info['Player_']=='Clifford Robinson')&(player_info['Year']==1994),'all_star'] = 1 
# Anfernee Hardway 1995,1996,1997,1998 - 4
player_info.loc[(player_info['Player_']=='Anfernee Hardaway')&(player_info['Year'].isin([1995,1996,1997,1998])),'all_star'] = 1 
# Nick Van 1998 - 1
player_info.loc[(player_info['Player_']=='Nick Van')&(player_info['Year']==1998),'all_star'] = 1 

## LAG all_star BACK 1 YEAR
# Index for each players
player_info.sort_values(['Player_','Year'],inplace=True)
player_info['Index_p'] = player_info.groupby(['Player_']).cumcount()
#player_info[['Player_','Year','Index_p']]
player_info_copy = player_info[['Player_','Index_p','all_star','Year']].copy()
player_info_copy['Index_p'] = player_info_copy['Index_p']  - 1
player_info_copy.columns = ['Player_2','Index_p2','all_star_ny','Year_copy']
player_info_copy.info()
player_info_lagged = pd.merge(player_info,player_info_copy,how='left',left_on=['Player_','Index_p'],right_on=['Player_2','Index_p2'])
player_info_lagged.info()
player_info_lagged[player_info_lagged['Year'] < 1984].info()
867-832
# 35 values of all star were removed, because:
# lost 24 all star in 1983
# 11 missing values: because players were voted in All Star for their 1st year according to data 
ma = pd.merge(player_info,player_info_copy,how='right',left_on=['Player_','Index_p'],right_on=['Player_2','Index_p2'])
ma.loc[ma['Player_'].isnull(),['Player_2','Index_p2','Year_copy','all_star_ny']][(ma['all_star_ny']==1)&
       (ma['Year_copy']>1983)]
#           Player_2        Index_p2  Year_copy  all_star_ny
#12185     Blake Griffin        -1     2011.0          1.0
#12576    David Robinson        -1     1990.0          1.0
#12674   Dikembe Mutombo        -1     1992.0          1.0
#12914        Grant Hill        -1     1995.0          1.0
#12946   Hakeem Olajuwon        -1     1985.0          1.0
#13741    Michael Jordan        -1     1985.0          1.0
#13916     Patrick Ewing        -1     1986.0          1.0
#13990     Ralph Sampson        -1     1984.0          1.0
#14244  Shaquille O'Neal        -1     1993.0          1.0
#14391        Tim Duncan        -1     1998.0          1.0
#14598          Yao Ming        -1     2003.0          1.0
player_info_lagged.loc[player_info_lagged['Player_']=='Blake Griffin',['Year','Player_','all_star','all_star_ny']]
player_info_lagged.loc[player_info_lagged['Player_']=='David Robinson',['Year','Player_','all_star','all_star_ny']]
player_info_lagged.loc[player_info_lagged['Player_']=='Dikembe Mutombo',['Year','Player_','all_star','all_star_ny']]
player_info_lagged.loc[player_info_lagged['Player_']=='Grant Hill',['Year','Player_','all_star','all_star_ny']]
player_info_lagged.loc[player_info_lagged['Player_']=='Hakeem Olajuwon',['Year','Player_','all_star','all_star_ny']]
player_info_lagged.loc[player_info_lagged['Player_']=='Michael Jordan',['Year','Player_','all_star','all_star_ny']]
player_info_lagged.loc[player_info_lagged['Player_']=='Patrick Ewing',['Year','Player_','all_star','all_star_ny']]
player_info_lagged.loc[player_info_lagged['Player_']=='Ralph Sampson',['Year','Player_','all_star','all_star_ny']]
player_info_lagged.loc[player_info_lagged['Player_']=="Shaquille O'Neal",['Year','Player_','all_star','all_star_ny']]
player_info_lagged.loc[player_info_lagged['Player_']=="Tim Duncan",['Year','Player_','all_star','all_star_ny']]
player_info_lagged.loc[player_info_lagged['Player_']=="Yao Ming",['Year','Player_','all_star','all_star_ny']]

#Add all star list from 2018
#player_info_lagged['Year'].max()
asr_18 = all_star_list.loc[all_star_list['Year'] > 2017,'Player']
len(asr_18)
player_info_lagged.loc[(player_info_lagged['Player_'].isin(asr_18))&(player_info_lagged['Year']>2016),'all_star_ny'] = 1
player_info_lagged.info()
832+28 #860 = 860 non-null all_star_ny

#Variables selection:
vabs_final = ['Year','Pos','G','MP','Age','PTS','FG','FG%','2P','2P%','3P','3P%',
                  'FT','FT%','AST','AST%','BLK','BLK%',
                  'DRB','DRB%','ORB','ORB%','STL','STL%',
                  'TOV','TOV%','PF','all_star_ny']
player_lagged_final = player_info_lagged[vabs_final]
player_lagged_final.info() ## 14617 total observations for stats from Season 1982-1983 to 2016-2017 
# Remove 1998 Year because there was no all star in 1999
player_lagged_final.loc[player_lagged_final['Year'] == 1998,'all_star_ny'].isnull().sum()
(player_lagged_final['Year'] == 1998).sum()
(player_lagged_final.loc[player_lagged_final['Year'] == 1998,'all_star_ny'] == 1).sum()
player_lagged_final.loc[(player_lagged_final['Year'] == 1998)&(player_lagged_final['all_star_ny'] == 1),]
# there was special case of Micheal Jordan when he took a break between playing periods
player_lagged_final = player_lagged_final.loc[player_lagged_final['Year'] != 1998,:]
player_lagged_final.info() ## 14178 total observations, excluding 1998

## DATA CLEANING 
# Data distribution
player_lagged_final.hist(figsize=(9,9))
# most %stat are normally distributed while other stats are usually right-skewed, except G
# it is better to convert other stats to average value per G, remove G 

# Fill missing value for all_star_ny as 0
player_lagged_final['all_star_ny'].fillna(0,inplace=True)

# Missing values were all from %indicator as there was no attempt. These missing values were  
# filled by zero-value as we considered no attempt as no success
player_lagged_final.fillna(0,inplace=True)
player_lagged_final.info()

## ------------------------------------------ EXPORT DATA ------------------------------------- ##
player_lagged_final.to_csv(os.path.join(file_path_local,r'player_lagged_final.csv'),index=False)
## -------------------------------------------------------------------------------------------- ##


## ------------------------------------------ IMPORT DATA ------------------------------------- ##
player_lagged_final = pd.read_csv(file_path_local+'player_lagged_final.csv')
## -------------------------------------------------------------------------------------------- ##



## --------------------------------------- DATA MANIPULATION -------------------------------------##
## CONVERT TOTAL VALUES TO AVERAGE VALUE PER GAME
player_lagged_final['MP_avg'] = player_lagged_final['MP']/player_lagged_final['G'] # Average Minute per game
player_lagged_final['PTS_avg'] = player_lagged_final['PTS']/player_lagged_final['G'] # Average Point per game
player_lagged_final['FG_avg'] = player_lagged_final['FG']/player_lagged_final['G'] # Average Field Goal per game
player_lagged_final['2P_avg'] = player_lagged_final['2P']/player_lagged_final['G'] # Average 2P per game
player_lagged_final['3P_avg'] = player_lagged_final['3P']/player_lagged_final['G'] # Average 3P per game
player_lagged_final['FT_avg'] = player_lagged_final['FT']/player_lagged_final['G'] # Average FT per game
player_lagged_final['AST_avg'] = player_lagged_final['AST']/player_lagged_final['G'] # Average AST per game
player_lagged_final['BLK_avg'] = player_lagged_final['BLK']/player_lagged_final['G'] # Average BLK per game
player_lagged_final['DRB_avg'] = player_lagged_final['DRB']/player_lagged_final['G'] # Average DRB per game
player_lagged_final['ORB_avg'] = player_lagged_final['ORB']/player_lagged_final['G'] # Average ORB per game
player_lagged_final['STL_avg'] = player_lagged_final['STL']/player_lagged_final['G'] # Average STL per game
player_lagged_final['TOV_avg'] = player_lagged_final['TOV']/player_lagged_final['G'] # Average TOV per game
player_lagged_final['PF_avg'] = player_lagged_final['PF']/player_lagged_final['G'] # Average PF per game
player_lagged_final[['MP_avg','PTS_avg','FG_avg','2P_avg','3P_avg','FT_avg',
                     'AST_avg','BLK_avg','DRB_avg','ORB_avg','STL_avg','TOV_avg','PF_avg']].hist(figsize=(9,9))

## SPLIT TRAIN-TEST DATA
## Split train-test sets as ratio of 80-20, stratied by Year 
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=25)
for train_index, test_index in split.split(player_lagged_final,player_lagged_final['Year']):
    strat_train_set = player_lagged_final.iloc[train_index]
    strat_test_set = player_lagged_final.iloc[test_index]

## IMBALANCE DATA SOLUTION
# Resampling
# OVERSAMPLING -----------------------------------------------------------------------------------     
from sklearn.utils import resample
# Separate the all-star and non-all-star
all_star = strat_train_set[strat_train_set['all_star_ny']==1] 
all_star.info()
non_all_star = strat_train_set[strat_train_set['all_star_ny']!=1] 
non_all_star.info()
# upsampling
all_star_upsampled = resample(all_star,replace=True,n_samples=len(non_all_star),random_state=25) 
len(all_star_upsampled)
all_star_upsampled.info()
# combining
train_upsampled = pd.concat([non_all_star,all_star_upsampled])
train_upsampled.info()

## DATA PROCESSING ------------------------------------------------------------------------------- 
## DATA SELECTION 
## Label variable
y_train = train_upsampled[['all_star_ny']] 

## Predictor variable
# Numerical variables
num_vab = ['G','MP_avg','Age','PTS_avg','FG_avg','FG%','2P_avg','2P%','3P_avg','3P%',
            'FT_avg','FT%','AST_avg','AST%','BLK_avg','BLK%','DRB_avg','DRB%','ORB_avg',
            'ORB%','STL_avg','STL%','TOV_avg','TOV%','PF_avg']
cat_vab = ['Pos']

class vab_select(BaseEstimator,TransformerMixin):
    def __init__(self,feature_names):
        self._feature_names = feature_names
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        return X[self._feature_names]
    
# Categorical pipeline for positions
train_upsampled['Pos'].value_counts()     
train_upsampled['PF'] = 0  
train_upsampled.loc[train_upsampled['Pos']=='PF','PF'] = 1  
train_upsampled['C'] = 0  
train_upsampled.loc[train_upsampled['Pos']=='C','C'] = 1  
train_upsampled['SG'] = 0  
train_upsampled.loc[train_upsampled['Pos']=='SG','SG'] = 1  
train_upsampled['PG'] = 0
train_upsampled.loc[train_upsampled['Pos']=='PG','PG'] = 1  
train_upsampled['SF'] = 0
train_upsampled.loc[train_upsampled['Pos']=='SF','SF'] = 1  
train_upsampled.loc[train_upsampled['Pos']=='PF-C',['PF','C']] = 1  
train_upsampled.loc[train_upsampled['Pos']=='SG-PG',['SG','PG']] = 1  
train_upsampled.loc[train_upsampled['Pos']=='SF-SG',['SF','SG']] = 1  
train_upsampled.loc[train_upsampled['Pos']=='PG-SG',['PG','SG']] = 1  
train_upsampled.loc[train_upsampled['Pos']=='C-PF',['PF','C']] = 1  
train_upsampled.loc[train_upsampled['Pos']=='SG-SF',['SG','SF']] = 1  
train_upsampled.loc[train_upsampled['Pos']=='PF-SF',['PF','SF']] = 1  
train_upsampled.loc[train_upsampled['Pos']=='SF-PF',['PF','SF']] = 1  
train_upsampled.loc[train_upsampled['Pos']=='SG-PF',['PF','SG']] = 1  
train_upsampled.loc[train_upsampled['Pos']=='C-SF',['C','SF']] = 1  
train_upsampled.loc[train_upsampled['Pos']=='PG-SF',['PG','SF']] = 1  

class position_dummy(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        X['PF'] = 0  
        X.loc[X['Pos']=='PF','PF'] = 1  
        X['C'] = 0  
        X.loc[X['Pos']=='C','C'] = 1  
        X['SG'] = 0  
        X.loc[X['Pos']=='SG','SG'] = 1  
        X['PG'] = 0
        X.loc[X['Pos']=='PG','PG'] = 1  
        X['SF'] = 0
        X.loc[X['Pos']=='SF','SF'] = 1  
        X.loc[X['Pos']=='PF-C',['PF','C']] = 1  
        X.loc[X['Pos']=='SG-PG',['SG','PG']] = 1  
        X.loc[X['Pos']=='SF-SG',['SF','SG']] = 1  
        X.loc[X['Pos']=='PG-SG',['PG','SG']] = 1  
        X.loc[X['Pos']=='C-PF',['PF','C']] = 1  
        X.loc[X['Pos']=='SG-SF',['SG','SF']] = 1  
        X.loc[X['Pos']=='PF-SF',['PF','SF']] = 1  
        X.loc[X['Pos']=='SF-PF',['PF','SF']] = 1  
        X.loc[X['Pos']=='SG-PF',['PF','SG']] = 1  
        X.loc[X['Pos']=='C-SF',['C','SF']] = 1  
        X.loc[X['Pos']=='PG-SF',['PG','SF']] = 1  
        return np.array(X[['PG','SG','SF','PF','C']])


## DATA PIPELINE
# Numerical pipeline
num_pipeline = Pipeline([('feature_seLect',vab_select(num_vab)),
                         ('data_scaling',MinMaxScaler())])        
num_train = num_pipeline.fit_transform(train_upsampled)  ## Numerical predictor variables
num_train.shape
# Categorical pipeline
cat_vab = ['Pos']
cat_pipeline = Pipeline([('feature_seLect',vab_select(cat_vab)),
                         ('dummy_positions',position_dummy())])        
# Feature Union
full_pipeline = FeatureUnion(transformer_list=[('numeric_pip',num_pipeline),
                                                  ('cat_pip',cat_pipeline)])
x_train = full_pipeline.fit_transform(train_upsampled)
x_train.shape


## MODELING ------------------------------------------------------------------------------ 
## Metrics
# Cross Validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
# Confusion Matrix
from sklearn.metrics import confusion_matrix
# Precision, Recall, F1 scores 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

## Candidate models
## SDC Classifier
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=25)
y_train_pred_sgdclf = cross_val_predict(sgd_clf,x_train,y_train,cv=5) 
# confusion matrix
confusion_matrix(y_train,y_train_pred_sgdclf)
#array([[9459, 1199],
#       [ 736, 9922]], dtype=int64)
# precision 
(9922)/(9922+1199) #0.8921
precision_score(y_train,y_train_pred_sgdclf)
# recall
(9922)/(736+9922) #0.9309
recall_score(y_train,y_train_pred_sgdclf)
# f1 
f1_score(y_train,y_train_pred_sgdclf) #0.9112
# ROC curve 
roc_auc_score(y_train,y_train_pred_sgdclf) #0.9092


## Logistic Regression
from sklearn.linear_model import LogisticRegression 
log_reg = LogisticRegression(random_state=25,solver='lbfgs') 
log_reg.fit(x_train,y_train)
y_train_pred_logreg = cross_val_predict(log_reg,x_train,y_train,cv=5)
# probability
y_train_proba_logreg = cross_val_predict(log_reg,x_train,y_train,cv=5,method='predict_proba')
y_train_proba_logreg
# confusion matrix
confusion_matrix(y_train,y_train_pred_logreg)
#array([[9485, 1173],
#       [ 726, 9932]], dtype=int64)
# precision
precision_score(y_train,y_train_pred_logreg) #0.8944
# recall
recall_score(y_train,y_train_pred_logreg) #0.9319
# f1
f1_score(y_train,y_train_pred_logreg) #0.9127
# ROC curve
roc_auc_score(y_train,y_train_pred_logreg) #0.9109


## Support Vector Machine: regularize the C for Hard-Soft margin classification
# Linear SVC
from sklearn.svm import LinearSVC
linearsvm_clf = LinearSVC(C=1,loss='hinge')
y_train_pred_logreg = cross_val_predict(linearsvm_clf,x_train,y_train,cv=5)
confusion_matrix(y_train,y_train_pred_logreg)
precision_score(y_train,y_train_pred_logreg) #0.8851
recall_score(y_train,y_train_pred_logreg) #0.9429
f1_score(y_train,y_train_pred_logreg) #0.9131
roc_auc_score(y_train,y_train_pred_logreg) #0.9103
# Nonlinear SVC
#from sklearn.svm import SVC
#polysvm = SVC(kernel='poly',degree=10,coef0=100,C=1)
#y_train_pred_polysvm = cross_val_predict(polysvm,x_train,y_train,cv=5) 


## Random Forest
from sklearn.ensemble import RandomForestClassifier 
rf_clf = RandomForestClassifier(random_state=25)
y_train_pred_rfclf = cross_val_predict(rf_clf,x_train,y_train,cv=5)
confusion_matrix(y_train,y_train_pred_rfclf)
precision_score(y_train,y_train_pred_rfclf) #0.9797
recall_score(y_train,y_train_pred_rfclf) #1.0
f1_score(y_train,y_train_pred_rfclf) #0.9897
roc_auc_score(y_train,y_train_pred_rfclf) #0.9896


## AdaBoost
from sklearn.ensemble import AdaBoostClassifier
ab_clf = AdaBoostClassifier(random_state=25)
y_train_pred_abc = cross_val_predict(ab_clf,x_train,y_train,cv=5)
confusion_matrix(y_train,y_train_pred_abc)
precision_score(y_train,y_train_pred_abc) #0.903
recall_score(y_train,y_train_pred_abc) #0.9377
f1_score(y_train,y_train_pred_abc) #0.92
roc_auc_score(y_train,y_train_pred_abc) #0.9185


## Gradient Boosting Machine
from sklearn.ensemble import GradientBoostingClassifier 
gb_clf = GradientBoostingClassifier(random_state=25)
y_train_pred_gbclf = cross_val_predict(gb_clf,x_train,y_train,cv=5)
confusion_matrix(y_train,y_train_pred_gbclf)
precision_score(y_train,y_train_pred_gbclf) #0.912
recall_score(y_train,y_train_pred_gbclf) #0.9811
f1_score(y_train,y_train_pred_gbclf) #0.9453
roc_auc_score(y_train,y_train_pred_gbclf) #0.9432


## Random forest has an extremly superior performance comparered to the other models
## even with the 2nd best model Gradient Boosting Machine where its precision score is 0.068 higher,
## its recall score is 0.019 higher, its f1 score is 0.044 higher and its ROC AUC score is 0.0464 higher     
## We try different parameters to find the most suitable parameters for Random forest model  
rf_clf
