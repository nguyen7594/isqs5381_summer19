# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 21:07:02 2019

@author: Nguyen7594
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### IMPORT CSV FILES
### CONTAIN FROM (1982-1983)-(2016-2017), YEAR VAB WILL BE ASSIGNED AS THE LAST YEAR OF THAT SEASON
## YEAR - SEASONS_STATS THE LAST YEAR OF THAT SEASON
## YEAR - ALL_STAR_LIST THE FIRST YEAR OF THAT SEASON
file_1 = 'Seasons_Stats.csv'
file_2 = 'NBA_All_Star_Games.csv'
file_3 = 'NBA_All_Star_Games_addition.csv'
# local file load
file_path = 'C:/Users/nguye/Documents/TTU/5381/nba/'
season_stat = pd.read_csv(file_path+file_1)
season_stat.info()
all_star_list = pd.read_csv(file_path+file_2)
all_star_list.info()
all_star_list_2 = pd.read_csv(file_path+file_3)
all_star_list_2.info()
# github load
file_path = 'https://raw.githubusercontent.com/nguyen7594/isqs5381_summer19nn/master/'
def file_import(file_name,FILE_PATH=file_path):
    csv_path = os.path.join(FILE_PATH,file_name)
    return pd.read_csv(csv_path)    
season_stat = file_import(file_1)
season_stat.info()
all_star_list = file_import(file_2)
all_star_list.info()
all_star_list_2 = file_import(file_3)
all_star_list_2.info()



## MERGE 2 ALL_STAR FILES
all_star = pd.concat([all_star_list,all_star_list_2])
all_star.info()
# Year is changed to the last half of season to match the season_stat
all_star.Year = all_star.Year + 1
# remove 1999 all star game: error - year without all-star
all_star = all_star[all_star['Year'] != 1999]
# 895 vabs
all_star[all_star['Year'] == 2018].count()
# 2018: 28 obs => rest = 895-28 = 867 obs


### JOIN SEASONS_STATS WITH ALL_STAR FILE FOR BEING ALL_STAR OR NOT
## ALL_STAR SHORT_LIST
all_star['all_star'] = 1
all_star['all_star'].groupby(all_star['Year']).count()
## REMOVE NA VALUES AND * AT THE END FOR SEASON_STAT
season_stat.dropna(subset=['Player'],inplace=True)

def remove_(X):
    if X[-1] == '*':
        return X.strip()[:-1]
    else:
        return X.strip()
    
season_stat['Player_'] = season_stat['Player'].map(lambda x: remove_(x))
## REMOVE DUPLICATES: only keep TOT (team ~ 'Tm') stats record for players
season_stat.drop_duplicates(['Year','Player_'],inplace=True)
        
## JOIN ALL_STAR WITH SEASON_STATS
season_stat_ = pd.merge(season_stat,all_star,left_on=['Year','Player_'],right_on=['Year','Player'],how='left')
season_stat_.info()
# 853 obs => 14 missing values from all_star

# players would not be matchaed 
left_out_ = pd.merge(season_stat,all_star,left_on=['Year','Player_'],right_on=['Year','Player'],how='right')
left_out_.info()
# 14 not matched 
left_out_[(left_out_['Player_x'].isnull())&(left_out_['Year']!=2018)][['Player_y','Year']]

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
# double check the count:
season_stat_.info()
# fill others as 0
season_stat_['all_star'].fillna(0,inplace=True)






