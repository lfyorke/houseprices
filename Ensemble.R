library(knitr)
library(caret)
library(plyr)
library(dplyr)
library(xgboost)
library(ranger)
library(nnet)
library(Metrics)
library(ggplot2)
library(gbm)
packages <- c("knitr", "caret", "plyr", "dplyr", "xgboost", "ranger", "nnet", "Metrics", "ggplot2", "gbm")

new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

setwd("C:/Users/leo.yorke/Desktop/Projects/kaggle/houseprices")

train <- read.csv("train.csv", stringsAsFactors = FALSE)

test <- read.csv("test.csv", stringsAsFactors = FALSE)


# Features are from rpevious Boruta analysis
conf_feats <- c("MSSubClass","MSZoning","LotArea","LotShape","LandContour","Neighborhood",
             "BldgType","HouseStyle","OverallQual","OverallCond","YearBuilt",
             "YearRemodAdd","Exterior1st","Exterior2nd","MasVnrArea","ExterQual",
             "Foundation","BsmtQual","BsmtCond","BsmtFinType1","BsmtFinSF1",
             "BsmtFinType2","BsmtUnfSF","TotalBsmtSF","HeatingQC","CentralAir",
             "X1stFlrSF","X2ndFlrSF","GrLivArea","BsmtFullBath","FullBath","HalfBath",
             "BedroomAbvGr","KitchenAbvGr","KitchenQual","TotRmsAbvGrd","Functional",
             "Fireplaces","FireplaceQu","GarageType","GarageYrBlt","GarageFinish",
             "GarageCars","GarageArea","GarageQual","GarageCond","PavedDrive","WoodDeckSF",
             "OpenPorchSF","Fence")  

tent_feats <- c("Alley","LandSlope","Condition1","RoofStyle","MasVnrType","BsmtExposure",
                "Electrical","EnclosedPorch","SaleCondition")

rej_feats <- c("LotFrontage","Street","Utilities","LotConfig","Condition2","RoofMatl",
               "ExterCond","BsmtFinSF2","Heating","LowQualFinSF","BsmtHalfBath",
               "X3SsnPorch","ScreenPorch","PoolArea","PoolQC","MiscFeature","MiscVal",
               "MoSold","YrSold","SaleType")

PREDICTOR_ATTR <- c(conf_feats,tent_feats,rej_feats)


data_types <- sapply(PREDICTOR_ATTR, function(x){class(train[[x]])})
unique_data_types <- unique(data_types)

data_class_types <- lapply(unique_data_types, function(x){names(data_types[data_types == x])})
names(data_class_types) <- unique_data_types

set.seed(13)
data_folds <- createFolds(train$SalePrice, k=5)

process_features_1 <- function(df) {
  id <- df$Id
  if (class(df$SalePrice) != "NULL") {
    y <- log(df$SalePrice)
  } else {
    y <- NULL
  }


predictor_vars <- c(conf_feats, tent_feats)
predictors <- df[predictor_vars]

# for numeric set missing values to -1 for purposes
num_attr <- intersect(predictor_vars,data_class_types$integer)
for (x in num_attr){
  predictors[[x]][is.na(predictors[[x]])] <- -1
}


# for character  atributes set missing value
char_attr <- intersect(predictor_vars,data_class_types$character)
for (x in char_attr){
  predictors[[x]][is.na(predictors[[x]])] <- "*MISSING*"
  predictors[[x]] <- factor(predictors[[x]])
}

return(list(id=id,y=y,predictors=predictors))

}


L0FeatureSet1 <- list(train=process_features_1(train),
                      test=process_features_1(test))



process_features_2 <- function(df) {
  id <- df$Id
  if (class(df$SalePrice) != "NULL") {
    y <- log(df$SalePrice)
  } else {
    y <- NULL
  }
  
  
  predictor_vars <- c(conf_feats,tent_feats)
  
  predictors <- df[predictor_vars]
  
  # for numeric set missing values to -1 for purposes
  num_attr <- intersect(predictor_vars,data_class_types$integer)
  for (x in num_attr){
    predictors[[x]][is.na(predictors[[x]])] <- -1
  }
  
  # for character  atributes set missing value
  char_attr <- intersect(predictor_vars,data_class_types$character)
  for (x in char_attr){
    predictors[[x]][is.na(predictors[[x]])] <- "*MISSING*"
    predictors[[x]] <- as.numeric(factor(predictors[[x]]))
  }
  
  return(list(id=id,y=y,predictors=as.matrix(predictors)))
}

L0FeatureSet2 <- list(train=process_features_2(train),
                      test=process_features_2(test))


# helper function
# train model on one data fold

train_one_fold <- function(fold, feature_set) {
  
  fold_data <- list() #initialise a list
  
  # get fold specific cv data
  fold_data$predictors <- feature_set$train$predictors[fold, ]  #  Get the predictors
  fold_data$ID <- feature_set$train$id[fold] # get the id
  fold_data$y <- feature_set$train$y[fold] # get target variable
  
  # get training data for specific fold
  train_data <- list()
  train_data$predictors <- feature_set$train$predictors[-fold,]
  train_data$y <- feature_set$train$y[-fold]
  
  set.seed(825)
  
  fitted_mdl <- do.call(caret::train, c(list(x=train_data$predictors,y=train_data$y),
                                 CARET.TRAIN.PARMS,
                                 MODEL.SPECIFIC.PARMS,
                                 CARET.TRAIN.OTHER.PARMS))
  
  yhat <- predict(fitted_mdl,newdata = fold_data$predictors,type = "raw")
  
  score <- rmse(fold_data$y,yhat)
  
  ans <- list(fitted_mdl=fitted_mdl,
              score=score,
              predictions=data.frame(ID=fold_data$ID,yhat=yhat,y=fold_data$y))
  
  return(ans)
  
}


makeOneFoldTestPrediction <- function(fold,feature_set) {
  fitted_mdl <- fold$fitted_mdl
  
  yhat <- predict(fitted_mdl,newdata = feature_set$test$predictors,type = "raw")
  
  return(yhat)
}


# gbm model

# set caret training parameters
CARET.TRAIN.PARMS <- list(method="gbm")   

CARET.TUNE.GRID <-  expand.grid(n.trees=100, 
                                interaction.depth=10, 
                                shrinkage=0.1,
                                n.minobsinnode=10)

MODEL.SPECIFIC.PARMS <- list(verbose=0) #NULL # Other model specific parameters

# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(method="none",
                                 verboseIter=FALSE,
                                 classProbs=FALSE)

CARET.TRAIN.OTHER.PARMS <- list(trControl=CARET.TRAIN.CTRL,
                                tuneGrid=CARET.TUNE.GRID,
                                metric="RMSE")

# generate features for Level 1
gbm_set <- llply(data_folds,train_one_fold,L0FeatureSet1)

# final model fit
gbm_mdl <- do.call(caret::train,
                   c(list(x=L0FeatureSet1$train$predictors,y=L0FeatureSet1$train$y),
                     CARET.TRAIN.PARMS,
                     MODEL.SPECIFIC.PARMS,
                     CARET.TRAIN.OTHER.PARMS))

# CV Error Estimate
cv_y <- do.call(c,lapply(gbm_set,function(x){x$predictions$y}))
cv_yhat <- do.call(c,lapply(gbm_set,function(x){x$predictions$yhat}))
rmse(cv_y,cv_yhat)

test_gbm_yhat <- predict(gbm_mdl,newdata = L0FeatureSet1$test$predictors,type = "raw")
gbm_submission <- cbind(Id=L0FeatureSet1$test$id,SalePrice=exp(test_gbm_yhat))

# xgboost model

# set caret training parameters
CARET.TRAIN.PARMS <- list(method="xgbTree")   

CARET.TUNE.GRID <-  expand.grid(nrounds=800, 
                                max_depth=10, 
                                eta=0.03, 
                                gamma=0.1, 
                                colsample_bytree=0.4, 
                                min_child_weight=1,
                                subsample = 1)



MODEL.SPECIFIC.PARMS <- list(verbose=0) #NULL # Other model specific parameters

# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(method="none",
                                 verboseIter=FALSE,
                                 classProbs=FALSE)

CARET.TRAIN.OTHER.PARMS <- list(trControl=CARET.TRAIN.CTRL,
                                tuneGrid=CARET.TUNE.GRID,
                                metric="RMSE")



# generate Level 1 features
xgb_set <- llply(data_folds,train_one_fold,L0FeatureSet2)

# final model fit
xgb_mdl <- do.call(caret::train,
                   c(list(x=L0FeatureSet2$train$predictors,y=L0FeatureSet2$train$y),
                     CARET.TRAIN.PARMS,
                     MODEL.SPECIFIC.PARMS,
                     CARET.TRAIN.OTHER.PARMS))

# CV Error Estimate
cv_y <- do.call(c,lapply(xgb_set,function(x){x$predictions$y}))
cv_yhat <- do.call(c,lapply(xgb_set,function(x){x$predictions$yhat}))
rmse(cv_y,cv_yhat)

cat("Average CV rmse:",mean(do.call(c,lapply(xgb_set,function(x){x$score}))))


test_xgb_yhat <- predict(xgb_mdl,newdata = L0FeatureSet2$test$predictors,type = "raw")
xgb_submission <- cbind(Id=L0FeatureSet2$test$id,SalePrice=exp(test_xgb_yhat))

# ranger model

# set caret training parameters
CARET.TRAIN.PARMS <- list(method="ranger")   

CARET.TUNE.GRID <-  expand.grid(mtry=2*as.integer(sqrt(ncol(L0FeatureSet1$train$predictors))), splitrule="variance")

MODEL.SPECIFIC.PARMS <- list(verbose=0,num.trees=500) #NULL # Other model specific parameters

# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(method="none",
                                 verboseIter=FALSE,
                                 classProbs=FALSE)

CARET.TRAIN.OTHER.PARMS <- list(trControl=CARET.TRAIN.CTRL,
                                tuneGrid=CARET.TUNE.GRID,
                                metric="RMSE")


# generate Level 1 features
rngr_set <- llply(data_folds,train_one_fold,L0FeatureSet1)

# final model fit
rngr_mdl <- do.call(caret::train,
                    c(list(x=L0FeatureSet1$train$predictors,y=L0FeatureSet1$train$y),
                      CARET.TRAIN.PARMS,
                      MODEL.SPECIFIC.PARMS,
                      CARET.TRAIN.OTHER.PARMS))

# CV Error Estimate
cv_y <- do.call(c,lapply(rngr_set,function(x){x$predictions$y}))
cv_yhat <- do.call(c,lapply(rngr_set,function(x){x$predictions$yhat}))
rmse(cv_y,cv_yhat)


cat("Average CV rmse:",mean(do.call(c,lapply(rngr_set,function(x){x$score}))))

test_rngr_yhat <- predict(rngr_mdl,newdata = L0FeatureSet1$test$predictors,type = "raw")
rngr_submission <- cbind(Id=L0FeatureSet1$test$id,SalePrice=exp(test_rngr_yhat))


#create predictions for use in nn

gbm_yhat <- do.call(c,lapply(gbm_set,function(x){x$predictions$yhat}))
xgb_yhat <- do.call(c,lapply(xgb_set,function(x){x$predictions$yhat}))
rngr_yhat <- do.call(c,lapply(rngr_set,function(x){x$predictions$yhat}))

# create Feature Set
L1FeatureSet <- list()

L1FeatureSet$train$id <- do.call(c,lapply(gbm_set,function(x){x$predictions$ID}))
L1FeatureSet$train$y <- do.call(c,lapply(gbm_set,function(x){x$predictions$y}))
predictors <- data.frame(gbm_yhat,xgb_yhat,rngr_yhat)
predictors_rank <- t(apply(predictors,1,rank))
colnames(predictors_rank) <- paste0("rank_",names(predictors))
L1FeatureSet$train$predictors <- predictors 

L1FeatureSet$test$id <- gbm_submission[,"Id"]
L1FeatureSet$test$predictors <- data.frame(gbm_yhat=test_gbm_yhat,
                                           xgb_yhat=test_xgb_yhat,
                                           rngr_yhat=test_rngr_yhat)



# set caret training parameters
CARET.TRAIN.PARMS <- list(method="nnet") 

CARET.TUNE.GRID <-  NULL  # NULL provides model specific default tuning parameters

# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(method="repeatedcv",
                                 number=5,
                                 repeats=1,
                                 verboseIter=FALSE)

CARET.TRAIN.OTHER.PARMS <- list(trControl=CARET.TRAIN.CTRL,
                                maximize=FALSE,
                                tuneGrid=CARET.TUNE.GRID,
                                tuneLength=7,
                                metric="RMSE")

MODEL.SPECIFIC.PARMS <- list(verbose=FALSE,linout=TRUE,trace=FALSE) #NULL # Other model specific parameters


set.seed(825)
l1_nnet_mdl <- do.call(caret::train,c(list(x=L1FeatureSet$train$predictors,y=L1FeatureSet$train$y),
                               CARET.TRAIN.PARMS,
                               MODEL.SPECIFIC.PARMS,
                               CARET.TRAIN.OTHER.PARMS))

l1_nnet_mdl

cat("Average CV rmse:",mean(l1_nnet_mdl$resample$RMSE),"\n")