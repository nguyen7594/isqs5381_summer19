library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(corrplot)
library(purrr)
library(stringr)
library(splitstackshape)
library(caret)
library(factoextra)
library(stringr)
library(forecast)
library(rpart)
library(randomForest)
library(neuralnet)
# Stochastic Gradient Boosting
library(gbm)
library(plyr)
library(e1071)
library(kernlab)
library(tibble)

## Import csv files
path = '~/TTU/5381/nba/isqs5381_summer19nn/src/'
all_set <- read_csv(paste0(path,'nba_8316.csv'))
set_explore <- read.csv(paste0(path,'nba_8317_explore.csv'))
names(set_explore)
## Overview the dataset ##
summary(set_explore[,-c(1,2,3,4)])

## Histogram of WS ##
set_explore %>%
  ggplot(aes(WS))+
  geom_histogram()
sd(set_explore$WS) ~~~ 3.092

## Histogram of all_star ##
set_explore %>%
  ggplot(aes(all_star))+
  geom_histogram()
sum(set_explore['all_star']==1)
sum(set_explore['all_star']==1)/nrow(set_explore)
sum(set_explore['all_star']==0)
sum(set_explore['all_star']==0)/nrow(set_explore)


#head(all_set)
#str(train_set)
#dim(all_set)
# Vabs chosen for x and y
colSums(is.na(all_set))
nrow(all_set)
vab_x <- c("Pos","G","MP","Age","PTS","FG","FG%",
           "2P","2P%","3P","3P%","FT","FT%","AST","AST%",
           "BLK","BLK%","DRB","DRB%","ORB","ORB%","STL","STL%",
           "TOV%","PF")
vab_y <- "WS2"
## split predictor and response variables
set_x <- all_set[,vab_x] 
set_y <- all_set[,vab_y]


## Data cleaning
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
#names(set_x)
summary(set_x)
summary(set_y)





### Split the training-valid-set
set.seed(1234)
df <- data.frame(all_set$Year,1:nrow(set_x))
df_index <-sample(1:nrow(set_x))
#head(df_index)
df <- df[df_index,]
#head(df)
trainvalid_ <-  createDataPartition(df$all_set.Year, times = 1, p = 0.8, list = FALSE)
# test data index
test_index <- df[-trainvalid_,]$X1.nrow.set_x.
#head(test_index)
# train-valid data index
trainvalid_index <- data.frame(index=df[trainvalid_,]$X1.nrow.set_x.,year=df[trainvalid_,]$all_set.Year)
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



## Correlation matrices 
#cor bw WS and predictor variables
cor_xy <- cor(train_ydf,train_xdf)
corrplot(cor_xy)
as.tibble(cor_xy) %>%
  gather('Features','Correlation') %>%
  arrange(Correlation)%>%
  ggplot(aes(reorder(Features,Correlation),Correlation))+
  geom_bar(stat='identity',aes(fill=Correlation))+
  coord_flip()+
  xlab('Features')

## Positions, Age do not seem important -> remove all positions
## PF misleading features -> remove PF
remove_unneed <- function(df){
  df[,-which(names(df) %in% c('pos.PG','pos.SG','pos.SF','pos.PF','pos.C','Age','PF'))]
}
train_xdf <- remove_unneed(train_xdf)
valid_xdf <- remove_unneed(valid_xdf)
test_xdf <- remove_unneed(test_xdf)


#cor bw predictor variables
cor_x <- cor(train_xdf)
corrplot(cor_x)
cor(train_xdf$X2P,train_xdf)
cor(train_xdf$FT,train_xdf)
cor(train_xdf$PF,train_xdf)
# High correlations bw 2P, FT, PF, PTS, FG, MP 
# Remove FG, FT, 2P
remove_highcor <- function(df){
  df[,-which(names(df) %in% c('FT','FG','X2P'))]
}
train_xdf <- remove_highcor(train_xdf)
valid_xdf <- remove_highcor(valid_xdf)
test_xdf <- remove_highcor(test_xdf)


##Visualization
vis_train <- all_set[train_index,]
vis_train$pri_pos <- sepPos[train_index,]$set_x.Pos_1 
#sum(is.na(vis_train$WS2))
#names(vis_train)
#head(vis_train[,c('PTS_AT','WS2')])

# PTS distribution
vis_train %>%
  ggplot(aes(PTS))+
  geom_histogram()

vis_train %>%
  ggplot(aes(pri_pos,PTS))+
  geom_boxplot()

# PTS v. WS2
vis_train %>%
  filter(PTS>15,MP>15)%>%
  ggplot(aes(PTS,WS2))+
  geom_jitter(aes(col=pri_pos),alpha=0.5)

  
# PTS/AT v. WS2 by positions
#2P
vis_train %>%
  filter(PTS>15,MP>15)%>%
  ggplot(aes(PTS,WS2))+
  geom_jitter(aes(size=`2P%`),alpha=0.4)+
  facet_wrap(~pri_pos)
#3P
vis_train %>%
  filter(PTS>15,MP>15)%>%
  ggplot(aes(PTS_AT,WS2))+
  geom_jitter(aes(size=`3P`),alpha=0.4)+
  facet_wrap(~pri_pos)
# 3P%  by positions
vis_train %>%
  filter(PTS>10,MP>15)%>%
  ggplot(aes(pri_pos,`3P%`))+
  geom_boxplot()
# AST v. PTS 
vis_train %>%
  filter(MP>15)%>%
  ggplot(aes(AST,PTS))+
  geom_jitter(aes(size=WS2),alpha=0.2)+
  facet_wrap(~pri_pos)


## Dimensional Reduction
# not suitable for dimensional reduction when there is low correlations
# between predictor variables
#train_xdf_dm <- train_xdf
#train_xdf_dm$TOV. <- max(train_xdf_dm$TOV.)- train_xdf_dm$TOV. 
#train_xdf_dm$PF <- max(train_xdf_dm$PF)- train_xdf_dm$PF 
#scaled_train_xdf_dm <- scale(train_xdf_dm)
#rownames(scaled_train_xdf_dm) <- str_c(all_set[train_index,]$Player_,all_set[train_index,]$Year,sep="-")
#head(scaled_train_xdf_dm)
# PCA
#x_pca <- princomp(scaled_train_xdf_dm[train_xdf_dm$PTS_AT>1,],cor=T)
#summary(x_pca,loadings=T)
#biplot(x_pca)
# K-mean clustering
#fviz_nbclust(scaled_train_xdf_dm, kmeans, method = "wss") +
#  geom_vline(xintercept = 3, linetype = 2,color='red')
#km <- kmeans(scaled_train_xdf_dm, centers = 3, nstart = 30) 
#fviz_cluster(km, data = scaled_train_xdf_dm,
#             ellipse.type = "norm", repel = FALSE, labelsize = 13
#)



#### MODEL ####
#names(train_xdf)
#names(train_ydf)
## Data scaling
norm_values <- preProcess(train_xdf,method=c('range')) 
train_norm_xdf <- predict(norm_values,train_xdf)
valid_norm_xdf <- predict(norm_values,valid_xdf)
test_norm_xdf <- predict(norm_values,test_xdf)

## WS2 ##
summary(train_ydf$WS2)
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#-1.600   0.500   2.200   3.038   4.600  20.400
hist(train_ydf$WS2)
sd(train_ydf$WS2)
# 3.16166
summary(valid_ydf$WS2)
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#-0.900   0.500   2.300   3.152   4.900  21.200 
sd(valid_ydf$WS2)
# 3.178022


#### MODEL CANDIDATES ####
#(1)
## LINEAR REGRESSION ##
#lm_nba <- lm(WS2 ~ .,data=cbind(train_ydf,train_norm_xdf))
lm_nba <- train(WS2 ~ .,cbind(train_ydf,train_norm_xdf),
                method='lm',
                trControl = trainControl(
                  method = 'cv',number = 10,
                  verboseIter = TRUE
                ))
#lm_nba <- train(WS2 ~ .,cbind(train_ydf,train_norm_xdf),
 #               method='lm',
  #              trControl = trainControl(
   #               method = 'repeatedcv',number = 10,
    #              verboseIter = TRUE, repeats=5
     #           ))
summary(lm_nba)
# test on valid data
pred_lm_valid <- predict(lm_nba,valid_norm_xdf)
accuracy(pred_lm_valid,valid_ydf$WS2)
#ME     RMSE      MAE MPE MAPE
#0.08963627 2.234972 1.674389 NaN  Inf


# (2)
## STOCHASTIC GRADIENT DESCENT ##
set.seed(1234)
sgd_nba <- train(WS2 ~ .,cbind(train_ydf,train_norm_xdf),
                         method='gbm',
                         trControl = trainControl(
                           method = 'cv',number = 10,
                           verboseIter = TRUE
                         ))
# test on valid data
pred_sgd_valid <- predict(sgd_nba,valid_norm_xdf)
accuracy(pred_sgd_valid,valid_ydf$WS2)
#ME     RMSE      MAE MPE MAPE
#0.1209268 2.182155 1.609092 NaN  Inf

# (3)
## REGRESSION TREE ##
set.seed(1234)
tree_nba <- train(WS2 ~ .,cbind(train_ydf,train_norm_xdf),
                method='rpart',
                trControl = trainControl(
                  method = 'cv',number = 10,
                  verboseIter = TRUE
                ))
# test on valid data
pred_tree_valid <- predict(tree_nba,valid_norm_xdf)
accuracy(pred_tree_valid,valid_ydf$WS2)
#ME     RMSE      MAE  MPE MAPE
#0.1091912 2.559957 1.882805 -Inf  Inf


# (4)
## SUPPORT VECTOR MACHINE ##
# Linear SVM #
set.seed(1234)
linear_svm_nba <- train(WS2 ~ .,cbind(train_ydf,train_norm_xdf),
                        method='svmLinear',
                        trControl = trainControl(
                          method = 'cv',number = 10,
                          verboseIter = TRUE
                        ))
# test on valid data
pred_linear_svm_valid <- predict(linear_svm_nba,valid_norm_xdf)
accuracy(pred_linear_svm_valid,valid_ydf$WS2)
#ME     RMSE      MAE MPE MAPE
#0.3513625 2.261586 1.638948 NaN  Inf

# (5)
## SUPPORT VECTOR MACHINE ##
# Non-Linear SVM #
set.seed(1234)
poly_svm_nba <- train(WS2 ~ .,cbind(train_ydf,train_norm_xdf),
                        method='svmPoly',
                        trControl = trainControl(
                          method = 'cv',number = 10,
                          verboseIter = TRUE
                        ))
# test on valid data
pred_polyr_svm_valid <- predict(poly_svm_nba,valid_norm_xdf)
accuracy(pred_polyr_svm_valid,valid_ydf$WS2)
#ME     RMSE      MAE MPE MAPE
#Test set 0.3134434 2.173623 1.557764 NaN  Inf
### Disadvantage: Take a lot time ###

# (6)
## RANDOM FOREST ##
set.seed(1234)
rf_nba <- train(WS2 ~ .,cbind(train_ydf,train_norm_xdf),
                  method='rf',importance=TRUE,
                  trControl = trainControl(
                    method = 'cv',number = 10,
                    verboseIter = TRUE
                  ))
varImpPlot(rf_nba,type=1)
# test on valid data
pred_rf_valid <- predict(rf_nba,valid_norm_xdf)
accuracy(pred_rf_valid,valid_ydf$WS2)
#ME     RMSE      MAE  MPE MAPE
#-0.08025988 2.112688 1.594136 -Inf  Inf



# (7)
## NEURAL NET ##
set.seed(1234)
nn_nba <- train(WS2 ~ .,cbind(train_ydf,train_norm_xdf),
                method='neuralnet',
                trControl = trainControl(
                  method = 'cv',number = 10,
                  verboseIter = TRUE
                ))





### COMPARE MODELS ###
results <- resamples(list(lm=lm_nba, sgd=sgd_nba))
summary(results,metric='RMSE')
bwplot(results)



### FINAL 3 MODELS ###

