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
library(gbm)
library(plyr)
library(e1071)
library(kernlab)
library(tibble)


## Import csv files ##
path = '~/TTU/5381/nba/isqs5381_summer19nn/src/'
# WS lagged created
all_set <- read_csv(paste0(path,'nba_8316.csv'))
# NO WS lagged
set_explore <- read_csv(paste0(path,'nba_8317_explore.csv'))

#### Overview the dataset ####
summary(set_explore[,-c(1,2,3,4)])
#nrow(all_set)

## All-star v. WS ##
cor(set_explore$all_star,set_explore$WS)

## All-star distribution ##
as_only <- set_explore[set_explore$all_star==1,]
summary(as_only)

## Histogram of WS ##
set_explore %>%
  ggplot(aes(WS))+
  geom_histogram()
sd(set_explore$WS) ~~~ 3.092

## Histogram of all_star ##
set_explore %>%
  ggplot(aes(all_star))+
  geom_histogram()
sum(set_explore['all_star']==1)/nrow(set_explore)
sum(set_explore['all_star']==0)/nrow(set_explore)

## Correlation ##
set_explore[is.na(set_explore)] = 0
cor_all <- cor(set_explore[,-c(1,2,3,4)],use = "everything")
corrplot(cor_all)


## Vabs chosen for x and y ##
vab_x <- c("Pos","G","MP","Age","PTS","FG","FG%",
           "2P","2P%","3P","3P%","FT","FT%","AST","AST%",
           "BLK","BLK%","DRB","DRB%","ORB","ORB%","STL","STL%",
           "TOV%","PF")
vab_y <- "WS2"
## split predictor and response variables
set_x <- all_set[,vab_x] 
set_y <- all_set[,vab_y]



#### Data cleaning ####
## Missing values ##
colSums(is.na(set_x))
# missing values are all in % categories, usually because they did not have any attempt/action 
# for that category. Thus, for these case, we assume no attempt/action is the same as o% success 
set_x[is.na(set_x)] <- 0

## Position ##
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
set_x <- data.frame(set_x[,-1],sepPos[,c('pos.PG','pos.SG','pos.SF','pos.PF','pos.C')])
set_x <- as.tibble(set_x)
names(set_x) <- c("G","MP","Age","PTS","FG","FG_perc",
                  "P2","P2_perc","P3","P3_perc","FT","FT_perc","AST","AST_perc",
                  "BLK","BLK_perc","DRB","DRB_perc","ORB","ORB_perc","STL","STL_perc",
                  "TOV%","PF",'pos.PG','pos.SG','pos.SF','pos.PF','pos.C') 



#### --------------------- Split the Traini-Valid-Test -------------------------- ####
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

## Split the data ## 
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
#nrow(set_x)
#nrow(test_xdf)+nrow(valid_xdf)+nrow(train_xdf)
hist(test_ydf$WS2)
sd(test_ydf$WS2)



#### -------------------------------- Visualization -------------------------------- ####
## Pair visualization ##
vis_train <- all_set[train_index,]
vis_train$pri_pos <- sepPos[train_index,]$set_x.Pos_1 

# PTS distribution #
vis_train %>%
  ggplot(aes(PTS))+
  geom_histogram()

vis_train %>%
  ggplot(aes(pri_pos,PTS))+
  geom_boxplot()

# PTS v. WS2 #
vis_train %>%
  filter(PTS>15,MP>15)%>%
  ggplot(aes(PTS,WS2))+
  geom_jitter(aes(col=pri_pos),alpha=0.5)


# PTS/AT v. WS2 by positions #
#2P
vis_train %>%
  filter(PTS>160,MP>160)%>%
  ggplot(aes(PTS,WS2))+
  geom_jitter(aes(size=`2P%`),alpha=0.4)+
  facet_wrap(~pri_pos)+
  ylab('WS')

#3P
vis_train %>%
  filter(PTS>160,MP>160)%>%
  ggplot(aes(PTS,WS2))+
  geom_jitter(aes(size=`3P%`),alpha=0.4)+
  facet_wrap(~pri_pos)+
  ylab('WS')

# 3P%  by positions #
vis_train %>%
  filter(PTS>160,MP>160)%>%
  ggplot(aes(pri_pos,`3P%`))+
  geom_boxplot()+
  xlab('Positions')

# DRB v. 2P #
vis_train %>%
  filter(MP>160,WS2>3,PTS>800)%>%
  ggplot(aes(pri_pos,DRB+ORB))+
  geom_boxplot()+
  xlab('Position')+
  ylab('Rebounds')

## Correlation matrices ## 
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


## ------------------------ DATA PROCESSING ------------------------ ##
### AVerage of cumulative variables
#train_xdf['PTS_avg'] <- train_xdf['PTS']/train_xdf['G']
#train_xdf['FT_avg'] <- train_xdf['FT']/train_xdf['G']
#rain_xdf['FG_avg'] <- train_xdf['FG']/train_xdf['G']
#train_xdf['P2_avg'] <- train_xdf['P2']/train_xdf['G']
#train_xdf['P3_avg'] <- train_xdf['P3']/train_xdf['G']
#train_xdf['MP_avg'] <- train_xdf['MP']/train_xdf['G']
#train_xdf['STL_avg'] <- train_xdf['STL']/train_xdf['G']
#train_xdf['AST_avg'] <- train_xdf['AST']/train_xdf['G']
#train_xdf['DRB_avg'] <- train_xdf['DRB']/train_xdf['G']
#train_xdf['ORB_avg'] <- train_xdf['ORB']/train_xdf['G']
#train_xdf['BLK_avg'] <- train_xdf['BLK']/train_xdf['G']
#train_xdf['PF_avg'] <- train_xdf['PF']/train_xdf['G']
### Feature selection
#c("G","MP","Age","PTS","FG","FG_perc",
#  "P2","P2_perc","P3","P3_perc","FT","FT_perc","AST","AST_perc",
#  "BLK","BLK_perc","DRB","DRB_perc","ORB","ORB_perc","STL","STL_perc",
#  "TOV%","PF",'pos.PG','pos.SG','pos.SF','pos.PF','pos.C') 


### Positions, Age, ORB do not seem important -> remove all positions
# PF misleading features -> remove PF
remove_unneed <- function(df){
  df[,-which(names(df) %in% c('pos.PG','pos.SG','pos.SF','pos.PF','pos.C','Age','ORB_perc'))]
}
train_xdf <- remove_unneed(train_xdf)
valid_xdf <- remove_unneed(valid_xdf)
test_xdf <- remove_unneed(test_xdf)

## cor bw predictor variables ##
cor_x <- cor(train_xdf)
corrplot(cor_x)
cor(train_xdf$X2P,train_xdf)
cor(train_xdf$FT,train_xdf)
cor(train_xdf$PF,train_xdf)
# High correlations bw 2P, FT, PF, PTS, FG, MP 
# Remove FG, FT, 2P
remove_highcor <- function(df){
  df[,-which(names(df) %in% c('FT','FG','P2'))]
}
train_xdf <- remove_highcor(train_xdf)
valid_xdf <- remove_highcor(valid_xdf)
test_xdf <- remove_highcor(test_xdf)
names(train_xdf)

## Convert values to average rather than seasonal data ##



## Dimensional Reduction ##
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


#### DATA PREPARATION ####
## Data scaling ##
norm_values <- preProcess(train_xdf,method=c('range')) 
train_norm_xdf <- predict(norm_values,train_xdf)
valid_norm_xdf <- predict(norm_values,valid_xdf)
test_norm_xdf <- predict(norm_values,test_xdf)


#### MODEL ####
## WS2 ##
#summary(train_ydf$WS2)
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#-1.600   0.500   2.200   3.038   4.600  20.400
#hist(train_ydf$WS2)
#sd(train_ydf$WS2)
# 3.16166
#summary(valid_ydf$WS2)
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#-0.900   0.500   2.300   3.152   4.900  21.200 
#sd(valid_ydf$WS2)
# 3.178022


### MODEL CANDIDATES ###
#(1)
## LINEAR REGRESSION ##
#lm_nba <- lm(WS2 ~ .,data=cbind(train_ydf,train_norm_xdf))
lm_nba <- train(WS2 ~ .,cbind(train_ydf,train_norm_xdf),
                method='lm',
                trControl = trainControl(
                  method = 'cv',number = 10,
                  verboseIter = TRUE
                ))
#summary(lm_nba)
# test on valid data
pred_lm_valid <- predict(lm_nba,valid_norm_xdf)
lm_valid <- accuracy(pred_lm_valid,valid_ydf$WS2)
lm_valid


# (2)
## GRADIENT BOOSTING MACHINE ##
set.seed(1234)
sgd_nba <- train(WS2 ~ .,cbind(train_ydf,train_norm_xdf),
                         method='gbm',
                         trControl = trainControl(
                           method = 'cv',number = 10,
                           verboseIter = TRUE
                         ))
# test on valid data
pred_sgd_valid <- predict(sgd_nba,valid_norm_xdf)
sgd_valid <- accuracy(pred_sgd_valid,valid_ydf$WS2)
sgd_valid

resid_sgd_valid <- valid_ydf$WS2 - pred_sgd_valid
hist(resid_sgd_valid)
plot(sgd_nba)



# (3)
## REGRESSION TREE ##
#set.seed(1234)
#tree_nba <- train(WS2 ~ .,cbind(train_ydf,train_norm_xdf),
#                method='rpart',
#                trControl = trainControl(
#                  method = 'cv',number = 10,
#                  verboseIter = TRUE
#                ))
# test on valid data
#pred_tree_valid <- predict(tree_nba,valid_norm_xdf)
#rt_valid <- accuracy(pred_tree_valid,valid_ydf$WS2)
#rt_valid


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
lin_svm_valid <- accuracy(pred_linear_svm_valid,valid_ydf$WS2)
lin_svm_valid


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
summary(poly_svm_nba)
# test on valid data
pred_polyr_svm_valid <- predict(poly_svm_nba,valid_norm_xdf)
poly_svm_valid <- accuracy(pred_polyr_svm_valid,valid_ydf$WS2)
poly_svm_valid

resid_polyr_svm_valid <- valid_ydf$WS2 - pred_polyr_svm_valid
hist(resid_polyr_svm_valid)


# (6)
## RANDOM FOREST ##
set.seed(1234)
rf_nba <- train(WS2 ~ .,cbind(train_ydf,train_norm_xdf),
                  method='rf',importance=TRUE,
                  trControl = trainControl(
                    method = 'cv',number = 10,
                    verboseIter = TRUE
                  ))
plot(varImp(rf_nba,type=1))
# test on valid data
pred_rf_valid <- predict(rf_nba,valid_norm_xdf)
rf_valid <- accuracy(pred_rf_valid,valid_ydf$WS2)
rf_valid
summary(rf_nba)
plot(rf_nba)
resid_rf_valid <- valid_ydf$WS2 - pred_rf_valid
hist(resid_rf_valid)

# (7)
## NEURAL NET ##
set.seed(1234)
nn_nba <- train(WS2 ~ .,cbind(train_ydf,train_norm_xdf),
                method='neuralnet',
                trControl = trainControl(
                  method = 'cv',number = 10,
                  verboseIter = TRUE
                ))
pred_nn_valid <- predict(nn_nba,valid_norm_xdf)
nn_valid <- accuracy(pred_nn_valid,valid_ydf$WS2)
nn_valid


### EVALUATE MODELS ###
## Train data ## #remove: reg_tree=tree_nba, 
results <- resamples(list(linear_model=lm_nba, gradient_boosting=sgd_nba,
                          linear_svm=linear_svm_nba,
                          poly_svm=poly_svm_nba, random_forest=rf_nba,
                          neural_net=nn_nba))
summary(results,metric=c('RMSE','MAE'))
bwplot(results,metric=c('RMSE','MAE'))
## Valid data ##   'reg_tree',as.data.frame(rt_valid)$MAE,as.data.frame(rt_valid)$RMSE,
metrics_valid <- tribble(~Model,~MAE,~RMSE,
                'linear_model',as.data.frame(lm_valid)$MAE,as.data.frame(lm_valid)$RMSE,
                'gradient_boosting',as.data.frame(sgd_valid)$MAE,as.data.frame(sgd_valid)$RMSE,
                'linear_svm',as.data.frame(lin_svm_valid)$MAE,as.data.frame(lin_svm_valid)$RMSE,
                'poly_svm',as.data.frame(poly_svm_valid)$MAE,as.data.frame(poly_svm_valid)$RMSE,
                'random_forest',as.data.frame(rf_valid)$MAE,as.data.frame(rf_valid)$RMSE,
                'neural_net',as.data.frame(nn_valid)$MAE,as.data.frame(nn_valid)$RMSE)
metrics_valid %>%
  gather('Metrics','Value',-Model)%>%
  ggplot(aes(x=Model))+
  geom_point(aes(y=Value,col=Metrics),size=3)+
  theme(axis.text.x = element_text(angle=30))+
  geom_hline(yintercept=as.data.frame(poly_svm_valid)$MAE,col="red",linetype="dashed")+
  geom_hline(yintercept=as.data.frame(poly_svm_valid)$RMSE,col="blue",linetype="dashed")


as.data.frame(metrics_valid)



### MODEL TUNING ###
## Using train() ##
# (however, Shrinkage and n.minobsinnode was held constant at 0.1 and 10)
training_ydf <- rbind(train_ydf,valid_ydf)
training_norm_xdf <- rbind(train_norm_xdf,valid_norm_xdf)
set.seed(1234)  
sgd_tuning <- train(WS2 ~ .,cbind(training_ydf,training_norm_xdf),
                    method='gbm',tuneLength=20,
                    trControl = trainControl(
                      method = 'cv',number = 5,
                      verboseIter = TRUE
                    ))
sgd_tuning
plot(sgd_tuning)
summary(sgd_tuning)


## using gbm() ##
hyper_grid <- expand.grid(
  shrinkage = c(.01, .1, .3),
  interaction.depth = c(1,2,3,4,5),
  n.minobsinnode = c(5, 10, 15),
  bag.fraction = c(.65, .8, 1), 
  optimal_trees = 0,               # a place to dump results
  min_RMSE = 0                     # a place to dump results
)
nrow(hyper_grid)
for(i in 1:nrow(hyper_grid)) {
    # reproducibility
  set.seed(1234)
    # train model
  gbm.tune <- gbm(
    formula = WS2 ~ .,
    distribution = "gaussian",
    data = cbind(training_ydf,training_norm_xdf),
    n.trees = 1000,
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = hyper_grid$bag.fraction[i],
    cv.folds = 5,
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(gbm.tune$cv.error)
  hyper_grid$min_RMSE[i] <- sqrt(gbm.tune$cv.error[hyper_grid$optimal_trees[i]])
}

## Top 10 ##
hyper_grid %>%
  arrange(min_RMSE)%>%
  top_n(-10,wt=min_RMSE)


### Train final model ###
# for reproducibility
set.seed(1234)
# train GBM model
nba.fit.final <- gbm(
  formula = WS2 ~ .,
  distribution = "gaussian",
  data = cbind(training_ydf,training_norm_xdf),
  n.trees = 358,
  interaction.depth = 3,
  shrinkage = 0.1,
  n.minobsinnode = 15,
  bag.fraction = .80, 
  train.fraction = 1,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)
pred_training <- predict(nba.fit.final,n.trees = nba.fit.final$n.trees,training_norm_xdf)
pred_training_metrics <- accuracy(pred_training,training_ydf$WS2)
pred_training_metrics


### save model to disk ###
saveRDS(nba.fit.final,"~/TTU/5381/nba/isqs5381_summer19nn/nba.fit.finalModel.rds")
#read mode: readRDS("~/TTU/5381/nba/isqs5381_summer19nn/nba.fit.finalModel.rds")


### Features importance ###
par(mar = c(5, 5, 1, 1))
summary(
  nba.fit.final, 
  cBars = 10,
  method = relative.influence, # also can use permutation.test.gbm
  las = 1
)


## Explaination for why 3P is not relatively important in model ##
# 2P v. 3P 
nba_since_2013 <- set_explore[set_explore$Year>2012,]
sum(nba_since_2013$`2P`)/sum(nba_since_2013$`3P`)



#### FINAL EVALUATION ####
# evaluate final model on Test data
pred_test <- predict(nba.fit.final, n.trees = nba.fit.final$n.trees,test_norm_xdf)
pred_metrics_test <- accuracy(pred_test,test_ydf$WS2)
pred_metrics_test


#### EXTRACT THE DATASET ####
train_x = rbind(train_norm_xdf,valid_norm_xdf)
train_y = rbind(train_ydf,valid_ydf) 
# write train data
write_csv(train_x,'~/TTU/5381/nba/isqs5381_summer19nn/train_x.csv')
write_csv(train_y,'~/TTU/5381/nba/isqs5381_summer19nn/train_y.csv')
# write test data
write_csv(test_norm_xdf,'~/TTU/5381/nba/isqs5381_summer19nn/test_x.csv')
write_csv(test_ydf,'~/TTU/5381/nba/isqs5381_summer19nn/test_y.csv')


### SUPPORT VECTOR MACHINE ###
library(e1071)
library(kernlab)


