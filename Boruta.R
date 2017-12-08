library(caret)
library(data.table)
library(Boruta)
library(plyr)
library(dplyr)
library(pROC)


ID.VAR <- "Id"
TARGET.VAR <- "SalePrice"

sample.df <- read.csv("train.csv",stringsAsFactors = FALSE) #read the csv

candidate.features <- setdiff(names(sample.df), c(ID.VAR, TARGET.VAR)) #remove ID.VAR and TARGET.VAR

data.classes <- sapply(candidate.features, function(x){class(sample.df[[x]])}) #Get the datatypes of each variable, sapply the class fucntion to all list values

unique.classes <- unique(data.classes)

attr.data.types <- lapply(unique.classes,function(x){names(data.classes[data.classes==x])})
names(attr.data.types) <- unique.classes

response <- sample.df$SalePrice  # Retrieve the target

sample.df <- sample.df[candidate.features]  # Remove ID and target variables

for (x in attr.data.types$integer) {
  sample.df[[x]][is.na(sample.df[[x]])] <- -1       #  Set missing numerics to -1
}

for (x in attr.data.types$character){
  sample.df[[x]][is.na(sample.df[[x]])] <- "*MISSING*" # Set missing strings to *MISSING*
}


set.seed(13) # Set the random seed

bor.results <- Boruta(sample.df, response, maxRuns=101, doTrace = 0)
print(bor.results)