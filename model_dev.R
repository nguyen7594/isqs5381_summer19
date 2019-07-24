library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(corrplot)
library(purrr)

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
set_x 



## Data cleaning
# Average the total indicators by game 
#str(train_x)
#names(train_x)
ave_func <- function(df){
  ave_col <- c("MP","PTS","FG",
              "2P","3P","FT","AST",
              "BLK","DRB","ORB","STL","PF")
  for (i in ave_col){
    df[,i] <- df[,i]/df$G
  }
}
#head(train_x)
#Position converted
unique(train_x$Pos)
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
