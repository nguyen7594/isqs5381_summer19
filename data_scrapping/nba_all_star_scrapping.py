# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 07:43:51 2019

@author: Nguyen7594
"""

import requests as r
from bs4 import BeautifulSoup
import csv
import time
import numpy as np

##### ---------------------- ALL STAR LIST ----------------------------#####
### URL AND YEARS NEEDED FOR SCRAPPING Season 1950-51 to 2018-19: Years ranges from 1951-2019 
url = 'https://basketball.realgm.com/nba/allstar/game/rosters/'
yearss = []
for i in range(1951,2019):
    yearss.append(i)


### FILE FOLDER AND CSV FILE
file_path = 'C:\\Users\\nguye\\Documents\\TTU\\5381\\nba\\isqs5381_summer19nn\\src'
file_name = 'NBA_All_Star_Games_195051to20172018.csv'

### CHECK STATUS
def statuscheck(res):
    if res.status_code == 200:
        print('request is good')
    else:
        print('bad request, received code ' + str(res.status_code))

### GET PAGE
def contentget(yr):
    urlget = url + str(yr)
    res = r.get(urlget)
    statuscheck(res)
    soup = BeautifulSoup(res.content,'lxml')
    return soup.find_all('tr') 
    
### VABRIABLES TO BE SCRAPPED
vab = ['Player','Pos','Height','Weight','Team','Selection Type','Draft Status','Nationality']

### COMPLETE FILE LINK
file_path_complete = file_path + file_name

### SAVE OBSERVATIONS TO CSV FILE
with open(file_path_complete, 'w',newline = '') as file1:
    datawriter = csv.writer(file1, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    datawriter.writerow(['Year','Player','Pos','HT','WT','Team','Selection Type','NBA Draft Status','Nationality'])
    for yr in yearss:
        player_result = contentget(yr)
        for i in range(len(player_result)):
            info_list = []
            for v in vab:
                if player_result[i].find('td',attrs={'data-th':v}) == None:
                    exit
                else:    
                    info_list.append(player_result[i].find('td',attrs={'data-th':v}).text)
            if info_list == []:
                next
            else:
                info_list.insert(0,yr)     
                datawriter.writerow(info_list)    
        time.sleep(np.random.randint(30,60))        



