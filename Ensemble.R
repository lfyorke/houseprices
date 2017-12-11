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
