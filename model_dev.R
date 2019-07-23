library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(corrplot)
set.seed(1234)
## Import csv files
path = '~/TTU/5381/nba/isqs5381_summer19nn/'
train_set <- read_csv(paste0(path,'training_set.csv'))
#head(train_set)
#str(train_set)
valid_set <- read.csv(paste0(path,'valid_set.csv'))
#head(valid_set)
#str(valid_set)
test_set <- read.csv(paste0(path,'test_set.csv'))
#head(test_set)
#str(test_set)

### Training set
## split predictor and response variables
# head(train_set$`2P%`)
train_x <- train_set[,4:29] 
train_y <- train_set[,30]

## Data cleaning
# Average the total indicators by game 
ave_value <- function(x){
  x/train_x$G
}
#str(train_x)
#names(train_x)
conv_col <- c(3,5,6,8,10,12,14,16,18,20,22,24,26)
for (i in conv_col){
  train_x[,i] <- ave_value(train_x[,i])
}
head(train_x)
cor_xy <- cor(train_y,train_x[,-1],use='pairwise.complete.obs')
cor_xy
cor_x <- cor(train_x[,-1],use='pairwise.complete.obs')
cor_x
corrplot(cor_x)
