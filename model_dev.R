library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(corrplot)
library(purrr)
library(stringr)

set.seed(1234)
## Import csv files
path = '~/TTU/5381/nba/isqs5381_summer19nn/'
all_set <- read_csv(paste0(path,'nba_8716.csv'))
#head(all_set)
#str(train_set)
dim(all_set)
# Vabs chosen for x and y
vab_x <- c("Pos","G","MP","Age","PTS","FG","FG%",
           "2P","2P%","3P","3P%","FT","FT%","AST","AST%",
           "BLK","BLK%","DRB","DRB%","ORB","ORB%","STL","STL%",
           "TOV%","PF")
vab_y <- "WS2"
## split predictor and response variables
set_x <- all_set[,vab_x] 
set_y <- all_set[,vab_y]


## Data cleaning
# Average the total indicators by game 
#str(train_x)
#names(train_x)
ave_col <- c("MP","PTS","FG",
              "2P","3P","FT","AST",
              "BLK","DRB","ORB","STL","PF")

for (i in ave_col){
    set_x[,i] <- set_x[,i]/set_x$G
  }
 
head(set_x)
#head(train_x)

#Missing values
colSums(is.na(set_x))
# missing values are all in % categories, usually because they did not have any attempt/action 
# for that category. Thus, for these case, we assume no attempt/action is the same as o% success 
set_x[is.na(set_x)] <- 0

# Position 
A <- c('PG-SF','C')
str_split(A,'-')


#cor bw WS and predictor variables
cor_xy <- cor(train_y,train_x[,-1],use='pairwise.complete.obs')
cor_xy
corrplot(cor_xy)
#cor bw predictor variables
cor_x <- cor(train_x[,-1],use='pairwise.complete.obs')
cor_x
corrplot(cor_x)


##Visualization
ggplot(train_set,aes(PTS,WS2))+
  geom_jitter(aes(size=PF))

## Dimensional Reduction
