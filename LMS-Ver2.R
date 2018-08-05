### LMS - Understanding the data ###

### Loading required packages ###
library(e1071)
library(randomForest)
library(xgboost)
library(caret)
library(DMwR)
library(Matrix)

### Setting working directory ###
filepath <- c("/Users/nkaveti/Documents/Kaggle/Last Man Standing")
setwd(filepath)

### Reading data into R ###
train <- read.csv("Input Data/Train.csv")
test <- read.csv("Input Data/Test.csv")

### Filling missing observations###
train[is.na(train$Number_Weeks_Used), "Number_Weeks_Used"] <- 28
test[is.na(test$Number_Weeks_Used), "Number_Weeks_Used"] <- 28

train$Season <- NULL
test$Season <- NULL

fac_cn <- c("Crop_Type", "Soil_Type", "Pesticide_Use_Category")
for(i in fac_cn){
  train[,i] <- as.factor(train[,i])
  test[,i] <- as.factor(test[,i])
}
train$Crop_Damage <- as.factor(train$Crop_Damage)

### Interactions ###
train$crop_pesticide <- interaction(train$Crop_Type, train$Pesticide_Use_Category)
train$insects_doses <- train$Estimated_Insects_Count * train$Number_Doses_Week
train$insects_week_used <- train$Estimated_Insects_Count * train$Number_Weeks_Used

train$insects_lag <- c(0, diff(train$Estimated_Insects_Count, lag = 5))

test$crop_pesticide <- interaction(test$Crop_Type, test$Pesticide_Use_Category)
test$insects_doses <- test$Estimated_Insects_Count * test$Number_Doses_Week
test$insects_week_used <- test$Estimated_Insects_Count * test$Number_Weeks_Used

test$insects_lag <- c(0, diff(test$Estimated_Insects_Count, lag = 5))


### xgboost ###
# train$Crop_Damage[train$Crop_Damage == 2] <- 1
train$Crop_Damage <- as.integer(train$Crop_Damage) - 1

num_folds <- 5
folds <- createFolds(as.factor(train$Crop_Damage), k = num_folds, list = FALSE)
train_pred <- c()
test_pred <- data.frame(matrix(0,nrow = nrow(test), ncol = num_folds*3 + 1))
# colnames(test_pred) <- c("ID", "Crop_Damage_1", "Crop_Damage_2", "Crop_Damage_3", "Crop_Damage_4", "Crop_Damage_5")
test_pred[,1] <- test$ID
colnames(test_pred)[1] <- c("ID")

params <- list(booster = "gbtree", eta = 0.01, gamma = 0.0, max_depth = 10, min_child_weight = 3, subsample = 0.6, colsample_bytree = 0.8, nthread = 4)

for(i in 1:num_folds){
  train_sparse <- sparse.model.matrix(~., data = train[!(folds == i), -c(1,9)])
  test_sparse <- sparse.model.matrix(~., data = train[folds == i, -c(1,9)])
  train_xgb <- xgb.DMatrix(train_sparse, label = train[!(folds == i), "Crop_Damage"])
  test_xgb <- xgb.DMatrix(test_sparse, label = train[folds == i, "Crop_Damage"])
  watchlist <- list(eval = test_xgb, train = train_xgb)
  xgb_model <- xgb.train(data = train_xgb, params = params, eval_metric = "merror", nrounds = 500, watchlist = watchlist, early.stop.round = 30, objective = "multi:softprob", num_class = 3)
  pre <- matrix(predict(xgb_model, test_xgb), nrow(test_xgb), 3, byrow = TRUE)
  train_pred <- rbind(train_pred, cbind(train[folds == i, c(1,9)],pre))
  te_pre <- matrix(predict(xgb_model, sparse.model.matrix(~., data = test[,-c(1)])), nrow(test), 3, byrow = TRUE)
  #te_pre <- (te_pre - min(te_pre))/(max(te_pre) - min(te_pre))
  test_pred[,((i-1)*3 + 2):(i*3 + 1)] <- te_pre
  colnames(test_pred)[((i-1)*3 + 2):(i*3 + 1)] <- c(paste0("Crop_Damage_0_",i), paste0("Crop_Damage_1_",i), paste0("Crop_Damage_2_",i))
  cat("Completed fold ", i, "\n")
}

colnames(train_pred) <- c("ID", "Crop_Damage", "Crop_Damage_0", "Crop_Damage_1", "Crop_Damage_2")


result <- sapply(1:nrow(train_pred), FUN = function(x,data){return(which.max(data[x,c(3:5)]))},data = train_pred)
train_pred$Crop_Damage_pr <- result - 1

train_pred$Crop_Damage_pr[train_pred$Crop_Damage_0 > 0.4] <- 0

tab <- table(train_pred$Crop_Damage, train_pred$Crop_Damage_pr)
acc <- sum(diag(tab/sum(tab)))

test_prob <- c()
for(i in 1:3){
  ind <- seq(i+1,16,3)
  cat(ind, "\n")
  test_prob <- cbind(test_prob, rowMeans(test_pred[,ind]))
}

write.csv(test_prob, file = "test_prob.csv", row.names = FALSE)
test_prob <- as.data.frame(test_prob)
test_prob$ID <- test_pred$ID

