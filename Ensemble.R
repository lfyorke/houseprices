library(knitr)
library(caret)
library(plyr)
library(dplyr)
library(xgboost)
library(ranger)
library(nnet)
library(Metrics)
library(ggplot2)
packages <- c("knitr", "caret", "plyr", "dplyr", "xgboost", "ranger", "nnet", "Metrics", "ggplot2")

new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

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


data_types <- sapply(conf_feats, function(x){class(train[[x]])})
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






