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
library(factoextra)
library(stringr)
library(forecast)
library(rpart)
library(randomForest)
library(neuralnet)


set.seed(1234)
## Import csv files
path = '~/TTU/5381/nba/isqs5381_summer19nn/'
all_set <- read_csv(paste0(path,'nba_8716.csv'))
#head(all_set)
#str(train_set)
#dim(all_set)
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
 

# New variables
# Points Per field goal attempts
set_x$PTS_AT <- set_x$PTS/(set_x$FG/set_x$`FG%`)  
set_x <- set_x%>%
  select(-c(FG,PTS,`FG%`))
#head(set_x)

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
#cor bw predictor variables
cor_x <- cor(train_xdf)
corrplot(cor_x)


##Visualization
vis_train <- all_set[train_index,]
for (i in ave_col){
  vis_train[,i] <- vis_train[,i]/vis_train$G
}
vis_train$pri_pos <- sepPos[train_index,]$set_x.Pos_1 
vis_train$PTS_AT <- vis_train$PTS/(vis_train$FG/vis_train$`FG%`)  
#sum(is.na(vis_train$WS2))
#names(vis_train)
#head(vis_train[,c('PTS_AT','WS2')])

# PTS/AT distribution
vis_train %>%
  ggplot(aes(PTS_AT))+
  geom_histogram()+
  facet_wrap(~pri_pos)+
  xlim(c(0,2))

vis_train %>%
  ggplot(aes(pri_pos,PTS_AT))+
  geom_boxplot()

# PTS/AT v. WS2
vis_train %>%
  filter(PTS>15,MP>15)%>%
  ggplot(aes(PTS_AT,WS2))+
  geom_jitter(aes(size=FG,col=pri_pos),alpha=0.5)

  
# PTS/AT v. WS2 by positions
#2P
vis_train %>%
  filter(PTS>15,MP>15)%>%
  ggplot(aes(PTS_AT,WS2))+
  geom_jitter(aes(size=`2P`),alpha=0.4)+
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
train_xdf_dm <- train_xdf
train_xdf_dm$TOV. <- max(train_xdf_dm$TOV.)- train_xdf_dm$TOV. 
train_xdf_dm$PF <- max(train_xdf_dm$PF)- train_xdf_dm$PF 
scaled_train_xdf_dm <- scale(train_xdf_dm)
rownames(scaled_train_xdf_dm) <- str_c(all_set[train_index,]$Player_,all_set[train_index,]$Year,sep="-")
head(scaled_train_xdf_dm)
# PCA
x_pca <- princomp(scaled_train_xdf_dm[train_xdf_dm$PTS_AT>1,],cor=T)
summary(x_pca,loadings=T)
biplot(x_pca)
# K-mean clustering
fviz_nbclust(scaled_train_xdf_dm, kmeans, method = "wss") +
  geom_vline(xintercept = 3, linetype = 2,color='red')
km <- kmeans(scaled_train_xdf_dm, centers = 3, nstart = 30) 
fviz_cluster(km, data = scaled_train_xdf_dm,
             ellipse.type = "norm", repel = FALSE, labelsize = 13
)



#### MODEL ####
#names(train_xdf)
#names(train_ydf)

## Data scaling
norm_values <- preProcess(train_xdf,method=c('range')) 
train_norm_xdf <- predict(norm_values,train_xdf)
valid_norm_xdf <- predict(norm_values,valid_xdf)
## WS2 ##
summary(train_ydf$WS2)
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#-1.600   0.500   2.200   3.097   4.700  21.200 
hist(train_ydf$WS2)
sd(train_ydf$WS2)
# 3.231
summary(valid_ydf$WS2)
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#-1.200   0.500   2.300   3.079   4.700  19.000 
sd(valid_ydf$WS2)
# 3.204




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
# test on training data
pred_lm_train <- predict(lm_nba,train_norm_xdf)
accuracy(pred_lm_train,valid_ydf$WS2)
#ME    RMSE    MAE MPE MAPE
#-0.08077606 4.03593 3.0282 NaN  Inf
# test on valid data
pred_lm_valid <- predict(lm_nba,valid_norm_xdf)
accuracy(pred_lm_valid,valid_ydf$WS2)
#ME     RMSE     MAE MPE MAPE
#-0.02966197 2.174304 1.65519 NaN  Inf


## GRADIENT DESCENT ##






## REGRESSION TREE ##
set.seed(1234)
tree_nba <- train(WS2 ~ .,cbind(train_ydf,train_norm_xdf),
                method='rpart',
                trControl = trainControl(
                  method = 'cv',number = 10,
                  verboseIter = TRUE
                ))
# test on training data
pred_tree_train <- predict(tree_nba,train_norm_xdf)
accuracy(pred_tree_train,valid_ydf$WS2)
#ME     RMSE      MAE  MPE MAPE
#-0.06818844 3.768866 2.836188 -Inf  Inf
# test on valid data
pred_tree_valid <- predict(tree_nba,valid_norm_xdf)
accuracy(pred_tree_valid,valid_ydf$WS2)
#ME     RMSE    MAE  MPE MAPE
#-0.006433537 2.585309 1.9921 -Inf  Inf



## RANDOM FOREST ##
set.seed(1234)
rf_nba <- train(WS2 ~ .,cbind(train_ydf,train_norm_xdf),
                  method='rf',importance=TRUE,
                  trControl = trainControl(
                    method = 'cv',number = 10,
                    verboseIter = TRUE
                  ))
set.seed(1234)
rf_nba_2 <- randomForest(WS2~.,cbind(train_ydf,train_norm_xdf),importance=TRUE)
varImpPlot(rf_nba_2,type=1)
# test on training data
pred_rf_train <- predict(rf_nba,train_norm_xdf)
accuracy(pred_rf_train,valid_ydf$WS2)
#ME     RMSE      MAE MPE MAPE
#-0.06728697 4.228946 3.105733 NaN  Inf
# Without Cross-validation
pred_rf_valid <- predict(rf_nba_2,train_norm_xdf)
accuracy(pred_rf_valid,train_ydf$WS2)
#ME      RMSE       MAE  MPE MAPE
#-0.01893074 0.8743806 0.6472731 -Inf  Inf
# test on valid data
pred_rf_valid <- predict(rf_nba,valid_norm_xdf)
accuracy(pred_rf_valid,valid_ydf$WS2)
#ME     RMSE      MAE  MPE MAPE
#-0.08025988 2.112688 1.594136 -Inf  Inf
# Without Cross-validation
pred_rf_valid <- predict(rf_nba_2,valid_norm_xdf)
accuracy(pred_rf_valid,valid_ydf$WS2)
#ME     RMSE      MAE  MPE MAPE
#-0.07844664 2.109372 1.593878 -Inf  Inf


## NEURAL NET ##
set.seed(1234)
nn_nba <- train(WS2 ~ .,cbind(train_ydf,train_norm_xdf),
                method='neuralnet',
                trControl = trainControl(
                  method = 'cv',number = 10,
                  verboseIter = TRUE
                ))
set.seed(1234)
nn <- neuralnet(WS2 ~ .,data=cbind(train_ydf,train_norm_xdf))
plot(nn)                
pred <- compute(nn,train_norm_xdf)
accuracy(pred$net.result[,1],train_ydf$WS2)
#ME     RMSE      MAE  MPE MAPE
#1.341957e-06 2.104113 1.557092 -Inf  Inf
pred <- compute(nn,valid_norm_xdf)
accuracy(pred$net.result[,1],valid_ydf$WS2)
#ME     RMSE     MAE  MPE MAPE
#-0.02311338 2.090387 1.55887 -Inf  Inf


