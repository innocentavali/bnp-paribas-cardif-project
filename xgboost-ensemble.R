## ------------------------------------------------------------------------
rm(list=ls(all=TRUE)) 

## ------------------------------------------------------------------------
library(xgboost)
library(data.table)
`%+%` <- function(a, b) paste(a, b, sep="")

## ------------------------------------------------------------------------
start_time <- Sys.time()
set.seed(2016289)

## ------------------------------------------------------------------------
message("Reading the train and test data")
train_raw <- fread("train.csv",stringsAsFactors=FALSE, showProgress=FALSE, sep=",", na.strings = c("NA",""))
train_target <- train_raw$target
train_dat <- as.data.frame(train_raw[,-c(1,2),with = FALSE])
test_raw <- fread("test.csv",stringsAsFactors=FALSE, showProgress=FALSE, sep=",", na.strings = c("NA",""))
test_dat <- as.data.frame(test_raw[,-1,with = FALSE])


## ------------------------------------------------------------------------
all_data <- rbind(train_dat,test_dat)
all_data <- all_data[,names(all_data)!="v22"] #too many levels
options(na.action='na.pass')
all_data <- model.matrix(~.-1,all_data)
options(na.action='na.omit')

## ------------------------------------------------------------------------
train_dat <- all_data[1:nrow(train_dat),]
test_dat <- all_data[(nrow(train_dat)+1):nrow(all_data),]
xgtrain = xgb.DMatrix(data.matrix(train_dat), label = train_target, missing = NA)
xgtest = xgb.DMatrix(data.matrix(test_dat), missing = NA)

## ------------------------------------------------------------------------
message("Data loaded")
print(difftime( Sys.time(), start_time, units = 'sec'))

## ------------------------------------------------------------------------
message("Training XGBoost classifiers")

param0 <- list(
  # some generic, non specific params
  "objective"  = "binary:logistic"
  , "eval_metric" = "logloss"
  , "eta" = 0.05
  , "min_child_weight" = 1
  , "max_depth" = 10
  , "nthread" = 32
  #, "scale_pos_weight" = 1
  #, "gamma" = 0.3
  #, "alpha" = 1e-05
  #, "lambda" = 0.1
  , "colsample_bytree" = 0.8
  , "subsample" = 0.8
)

#number of ensmeble models
n = 10

message("\nCross validate XGBoost classifier")
print(difftime( Sys.time(), start_time, units = 'sec'))

ensemble_cv <- xgb.cv(data = xgtrain, params = param0, nround = 5000, nfold = 5, early.stop.round = 10, print.every.n = 20)
best <- min(ensemble_cv$test.logloss.mean)
bestIter <- which(ensemble_cv$test.logloss.mean==best)

ensemble_p <- matrix(0, nrow = nrow(test_dat), ncol = n)
ensemble_model <- vector("list", n) 
watchlist <- list('train' = xgtrain)

for (i in 1:n){
  set.seed(2016 + 289*i)
  message("\nTraining the "%+%i%+%"th XGBoost classifier")
  print(difftime( Sys.time(), start_time, units = 'sec'))
  ensemble_model[[i]] <- xgb.train(data = xgtrain, params = param0, nround = round(bestIter*1.5), watchlist = watchlist, print.every.n = 20)
  ensemble_p[,i] <- predict(ensemble_model[[i]], xgtest)
}

## ------------------------------------------------------------------------
save(ensemble_p, ensemble_model, ensemble_cv, file = "bnp-xgb-ohe-ensemble.RData")

## ------------------------------------------------------------------------
cat("Making predictions\n")
submission <- data.frame(ID=test_raw$ID,PredictedProb=apply(ensemble_p,1,mean))
write.csv(submission,"bnp-xgb-ohe-ensemble.csv",row.names=F, quote=F)
print( difftime( Sys.time(), start_time, units = 'min'))