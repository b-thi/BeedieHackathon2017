### Extreme Gradient Boost ###



#### Example 1

install.packages("drat")
drat::addRepo("dmlc")
install.packages("xgboost", repos = "http://dmlc.ml/drat/", type = "source")

library(xgboost)

# Use carets for real data
data(agaricus.train, package ='xgboost')
data(agaricus.test, package='xgboost')
train <- agaricus.train
test <- agaricus.test

dim(train$data)
dim(test$data)

# Need the matrix to be in a dgCMatrix format.  Xgboost works best when it is a sparse matrix.
class(train$data)
class(train$label)

# Basic Training

bstSparse <- xgboost(data = train$data,
                     label = train$label,
                     max.depth = 2,
                     eta = 1,
                     nthread = 2,
                     nround = 2,
                     objective = "binary:logistic")


# Parameter Variations

bstDense <- xgboost(data = as.matrix(train$data),
                    label = train$label,
                    max.depth = 2,
                    eta = 1,
                    nthread = 2,
                    nround = 2,
                    objective = "binary:logistic")

dtrain <- xgb.DMatrix(data = train$data, label = train$label)

bestDMatrix <- xgboost(data = dtrain,
                       max.depth = 2,
                       eta = 1,
                       nthread = 2,
                       nround = 2,
                       objective = "binary:logistic")


bst <- xgboost(data = dtrain,
               max.depth = 2,
               eta = 1,
               nthread = 2,
               nround = 2,
               objective = "binary:logistic",
               verbose = 1) # Print evaluation metrix, 0 = no print, 2 = extra information

# Prediction

pred <- predict(bst, test$data)
pred

prediction <- as.numeric(pred > 0.5)
prediction

# Measuring Model Performance

err <- mean(as.numeric(pred > 0.5) != test$label)
print(paste("test-error", err))


# Advanced Features

dtrain <- xgb.DMatrix(data = train$data, label = train$label)
dtest <- xgb.DMatrix(data = test$data, label = test$label)

# Both xgboost (simple) and xgb.train (advanced) functions train models.

# use watchlist parameter.  It is a list of xgb.DMatrix, each of them tagged with a name.
watchlist <- list(train = dtrain, test = dtest)

bst <- xgb.train(data = dtrain,
                 max.depth = 2,
                 eta = 1,
                 nthread = 2,
                 nround = 2, watchlist = watchlist,
                 eval.metric = "logloss",
                 eval.metric = "error",
                 objective = "binary:logistic")


# Linear Boosting
# XGBoost implements a second algorithm based on linear boosting.

bst <- xgb.train(data = dtrain,
                 booster = "gblinear",
                 max.depth = 2,
                 nthread = 2,
                 nround = 2,
                 watchlist = watchlist,
                 eval.metric = "error",
                 eval.metric = "logloss",
                 objective = "binary:logistic")

# We see here that linear boosting gets a slightly better performance metrics than decision trees based algorithm.

# Information Extraction

label = getinfo(dtest, "label")
pred <- predict(bst, dtest)
err <- as.numeric(sum(as.integer(pred > 0.5) != label))/length(label)
print(paste("test-error=", err))

# View feature importance/influence from the learnt model

importance_matrix <- xgb.importance(model = bst)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix)

xgb.dump(bst, with.stats = T)

xgb.plot.tree(model = bst)


### Example 2: Hyperparameter Tuning ######

library(caret)
library(xgboost)
library(readr)
library(dplyr)
library(tidyr)

setwd("C:/Users/Brad_/Documents/Data Science/Model Experiments/Extreme Gradient Boost")

# load in the training data
df_train = read_csv("cs-training.csv") %>%
  na.omit() %>%                                                                # listwise deletion 
  select(-`[EMPTY]`) %>%
  mutate(SeriousDlqin2yrs = factor(SeriousDlqin2yrs,                           # factor variable for classification
                                   labels = c("Failure", "Success")))

# xgboost fitting with arbitrary parameters
xgb_params_1 = list(
  objective = "binary:logistic",                                               # binary classification
  eta = 0.01,                                                                  # learning rate
  max.depth = 3,                                                               # max tree depth
  eval_metric = "auc"                                                          # evaluation/loss metric
)

# fit the model with the arbitrary parameters specified above
xgb_1 = xgboost(data = as.matrix(df_train %>%
                                   select(-SeriousDlqin2yrs)),
                label = df_train$SeriousDlqin2yrs,
                params = xgb_params_1,
                nrounds = 100,                                                 # max number of trees to build
                verbose = TRUE,                                         
                print.every.n = 1,
                early.stop.round = 10                                          # stop if no improvement within 10 trees
)

# cross-validate xgboost to get the accurate measure of error
xgb_cv_1 = xgb.cv(params = xgb_params_1,
                  data = as.matrix(df_train %>%
                                     select(-SeriousDlqin2yrs)),
                  label = df_train$SeriousDlqin2yrs,
                  nrounds = 100, 
                  nfold = 5,                                                   # number of folds in K-fold
                  prediction = TRUE,                                           # return the prediction using the final model 
                  showsd = TRUE,                                               # standard deviation of loss across folds
                  stratified = TRUE,                                           # sample is unbalanced; use stratified sampling
                  verbose = TRUE,
                  print.every.n = 1, 
                  early.stop.round = 10
)

# plot the AUC for the training and testing samples
xgb_cv_1$dt %>%
  select(-contains("std")) %>%
  mutate(IterationNum = 1:n()) %>%
  gather(TestOrTrain, AUC, -IterationNum) %>%
  ggplot(aes(x = IterationNum, y = AUC, group = TestOrTrain, color = TestOrTrain)) + 
  geom_line() + 
  theme_bw()



# Hyperparameter Tuning with grid search

searchGridSubCol <- expand.grid(subsample = c(0.5, 0.75, 1),
                                colsample_bytree = c(0.6, 0.8, 1))

ntrees <- 100

# Build a xgb.DMatrix object
DMMatrixTrain <- xgb.DMatrix(data = train$data, label = train$label)

rmseErrorsHyperparameters <- apply(searchGridSubCol, 1, function(parameterList){
  
  #Extrat Parameters to test
  currentSubsampleRate <- parameterList[["subsample"]]
  currentColsampleRate <- parameterList[["colsample_bytree"]]
  
  xgboostModelCV <- xgb.cv(data = DMMatrixTrain,
                           nrounds = ntrees,
                           nfold = 5,
                           showsd = TRUE,
                           metrics = "rmse",
                           verbose = TRUE,
                           eval.metric = "rmse",
                           objevtive = "reg:linear",
                           max.depth = 15,
                           eta = 2/ntrees,
                           subsample = currentSubsampleRate,
                           colsample_bytree = currentColsampleRate)
  
  xvalidationScores <- xgboostModelCV
  #Save rmse of the last iteration
  rmse <- tail(xvalidationScores$evaluation_log$test_rmse_mean, 1)
  
  return(c(rmse, currentSubsampleRate, currentColsampleRate))
})


rmseErrorsHyperparameters


# Example 3: Expand Grid

# set up the cross-validated hyper-parameter search
xgb_grid_1 = expand.grid(
  nrounds = 1000,
  eta = c(0.01, 0.001, 0.0001),
  max_depth = c(2, 4, 6, 8, 10),
  gamma = 1
)

# pack the training control parameters
xgb_trcontrol_1 = trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "all",                                                        # save losses across all models
  classProbs = TRUE,                                                           # set to TRUE for AUC to be computed
  summaryFunction = twoClassSummary,
  allowParallel = TRUE
)

# train the model for each parameter combination in the grid,
#   using CV to evaluate
xgb_train_1 = train(
  x = as.matrix(df_train %>%
                  select(-SeriousDlqin2yrs)),
  y = as.factor(df_train$SeriousDlqin2yrs),
  trControl = xgb_trcontrol_1,
  tuneGrid = xgb_grid_1,
  method = "xgbTree"
)

# scatter plot of the AUC against max_depth and eta
ggplot(xgb_train_1$results, aes(x = as.factor(eta), y = max_depth, size = ROC, color = ROC)) +
  geom_point() +
  theme_bw() +
  scale_size_continuous(guide = "none")













