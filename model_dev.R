library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(corrplot)
library(purrr)
library(stringr)
#install.packages('splitstackshape')
library(splitstackshape)
library(caret)


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
#unique(set_x$Pos)
pos_sep <- data.frame(all_set$Player_,set_x$Pos)
#head(pos_sep)
sepPos <- cSplit(pos_sep,'set_x.Pos',sep='-')
sepPos$pos.PG <- 0
sepPos$pos.SG <- 0
sepPos$pos.SF <- 0
sepPos$pos.PF <- 0
sepPos$pos.C <- 0
sepPos[(sepPos$set_x.Pos_1=='PG')|(sepPos$set_x.Pos_2=='PG'),'pos.PG']<-1
sepPos[(sepPos$set_x.Pos_1=='SG')|(sepPos$set_x.Pos_2=='SG'),'pos.SG']<-1
sepPos[(sepPos$set_x.Pos_1=='SF')|(sepPos$set_x.Pos_2=='SF'),'pos.SF']<-1
sepPos[(sepPos$set_x.Pos_1=='PF')|(sepPos$set_x.Pos_2=='PF'),'pos.PF']<-1
sepPos[(sepPos$set_x.Pos_1=='C')|(sepPos$set_x.Pos_2=='C'),'pos.C']<-1
#head(sepPos)
set_x <- data.frame(set_x[,-1],sepPos[,c('pos.PG','pos.SG','pos.SF','pos.PF','pos.C')])
names(set_x)

### Split the training-valid-set
set.seed(1234)
df <- data.frame(all_set$Year,1:nrow(set_x))
df_index <-sample(1:nrow(set_x))
#head(df_index)
df <- df[df_index,]
#head(df)
trainvalid_ <-  createDataPartition(df$all_set.Year, times = 1, p = 0.8, list = FALSE)
# test data index
test_index <- df[-train_x_index,]$X1.nrow.set_x.
#head(test_index)
# train-valid data index
trainvalid_index <- data.frame(index=df[train_x_index,]$X1.nrow.set_x.,year=df[train_x_index,]$all_set.Year)
#head(trainvalid_index)
train_ <- createDataPartition(trainvalid_index$year, times = 1, p = 0.8, list = FALSE)
# train index
train_index <- trainvalid_index[train_,]$index
#head(train_index)
# valid index
valid_index <- trainvalid_index[-train_,]$index
#head(valid_index)

## Split the data 
# train df
train_xdf <- set_x[train_index,] 
train_ydf <- set_y[train_index,]
#nrow(train_xdf)
# valid df
valid_xdf <- set_x[valid_index,] 
valid_ydf <- set_y[valid_index,]
#nrow(valid_xdf)
# test df
test_xdf <- set_x[test_index,] 
test_ydf <- set_y[test_index,]
#nrow(test_xdf)




#cor bw WS and predictor variables
cor_xy <- cor(train_ydf,train_xdf)
names(train_xdf)
cor_xy
corrplot(cor_xy)
#cor bw predictor variables
cor_x <- cor(train_x)
corrplot(cor_x)


##Visualization
vis_train <- all_set[train_index,]
for (i in ave_col){
  vis_train[,i] <- vis_train[,i]/vis_train$G
}

vis_train %>%
  filter(PTS>5,Pos=='PG')%>%
  ggplot(aes(`3P%`,WS2))+
  geom_jitter(aes(size=FG))



## Dimensional Reduction
