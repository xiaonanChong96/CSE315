# 1 Split the data into a training and a test set, 
# 2 pre-process the data, 
# 3 and build linear regression model, 
# robust linear model, (partial least squares model, ridge regression model, and elastic nets model) 
# for [predicting the fat content] of a sample. 

# For those models with tuning parameters, what are the optimal values of the tuning parameter(s)? 
# Which model has the best predictive ability? 
# Is any model significantly better or worse than the others?

# MODELS    | linear regression | robust linear | PLS     | redge regression | elastic nets |
# RMSE      | 2.307             | 6.868         | 1.528   | 3.589            | 
# R Square  | 0.968             | 0.732         | 0.986   | 0.930            | 
# MAE       | 1.737             | 4.943         | 1.085   | 2.623            | 

library(caret)
data(tecator)

#(the final purpose is to predict the percentage of fat, therefore extract only fat column)
fat <- endpoints[1:215,2]

#split the data into a training and test set
##set.seed(1)
##trainRows <- createDataPartition(fat, p=.80, list = FALSE)

##trainPredictors <- absorp[trainRows, ]
##trainClasses <- fat[trainRows]
##str(trainPredictors)

##testPredictors <- absorp[-trainRows, ]
##testClasses <- fat[-trainRows]
##str(testPredictors)

#(according to the documentation, the first 129 samples are used as training samples)
train<- absorp[1:129,]
test <- absorp[130:215,]
trainY <- endpoints[1:129,2]
testY <- endpoints[130:215,2]

##data preprocessing============================================================================
library(e1071)
skewValues <- apply(train, 2, skewness)
head(skewValues)

library(lattice)
histogram(skewValues)

library(caret)
trainP <- prcomp(trainPredictors,
                  center=TRUE,
                  scale.=TRUE)

#create the column names
tmp<-data.frame()
for (i in 1:100){
  tmp[i,1] <-paste("channel",toString(i))}
colnames(train)=c(tmp[,1])
colnames(test)=c(tmp[,1])
#transformation
transform <- preProcess(train, 
                        method = c("BoxCox", "center", "scale"))

#just check the pca result
transform2 <- preProcess(train, 
                  method = c("pca"))
pca <- predict(transform2, train)

#apply the transformation 
trainXtrans <- predict(transform, train)
testXtrans <- predict(transform, test)

## build linear regression models===============================================================
##simple least square
trainingData <- cbind(trainXtrans, trainY)
colnames(trainingData) # the last column called - trainY
lmFitAllPredictors <- lm( trainY~., data = as.data.frame(trainingData))
summary(lmFitAllPredictors) # RMSE: 1.074; R square: 0.998

lmPred1 <- predict(lmFitAllPredictors, as.data.frame(testXtrans))
head(lmPred1)
lmValues1 <- data.frame(obs = testY, pred = lmPred1)
defaultSummary(lmValues1) # RMSE 2.307; R square: 0.968; MAE: 1.737


#robust linear model
library(MASS)
rlmFitAllPredictors  <- rlm( trainY~., data = as.data.frame(trainingData))
#10-floder cross validation
ctrl <- trainControl(method = "cv", number = 10)
set.seed(100)
lmFit1 <- train(x = trainXtrans, y = trainY, method = "lm", trControl = ctrl)
lmFit1

#diagnos plots
xyplot(trainY ~ predict(lmFit1),type = c("p", "g"),xlab = "Predicted", ylab = "Observed")
xyplot(resid(lmFit1) ~ predict(lmFit1),type = c("p", "g"),xlab = "Predicted", ylab = "Residuals")


#partial least square
library(pls)
plsFit <- plsr(trainY ~ ., data = as.data.frame(trainingData))
predict(plsFit, testXtrans)

set.seed(100)
plsTune <- train(trainXtrans, trainY, method = "pls",tuneLength = 20,
                   trControl = ctrl,preProc = c("center", "scale"))
plsTune # the final model use 19 components 


#ridge regression model
library(elasticnet)
ridgeModel <- enet(x = trainXtrans, y = trainY, lambda = 0.001)
ridgePred <- predict(ridgeModel, newx = as.matrix(testXtrans),
                     s = 1, mode = "fraction",type = "fit")
head(ridgePred$fit)

#tune over the penalty
ridgeGrid <- data.frame(.lambda = seq(0, .1, length = 15))
set.seed(100)
ridgeRegFit <- train(trainXtrans, trainY,
                     method = "ridge",
                     ## Fir the model over many penalty values
                     tuneGrid = ridgeGrid,
                     trControl = ctrl,
                     ## put the predictors on the same scale
                     preProc = c("center", "scale"))
ridgeRegFit
                     
#elastic nets
library(elasticnet)
                     
                     