## Set working directory --------------------------------------------------------
getwd()

## Load libraries ---------------------------------------------------------------
library(caret)
library(ggplot2)
library(ROCR) #for plotting ROC curves.
library(MLTools)
library(randomForest)

## remove all epxisting variables--------------------------------------------------
rm(list=ls())



#-------------------------------------------------------------------------------------------------
#---------------------------------       TASKS       ---------------------------------------------
#-------------------------------------------------------------------------------------------------

# 1. Study the efect of ntree (nº of trees) in the problem at hand
# 2. Apply bagging and random forests to the temperature data 
#    to estimate SEASON from the rest

# # Read temperature dataset 
fdata = read.table("TemperatureDataSpain.dat", sep = "", header = TRUE, stringsAsFactors = FALSE)
str(fdata)
fdata$Y = as.factor(fdata$SEASON)
str(fdata)


## Divide the data into training and validation sets ---------------------------------------------------
set.seed(150) #For replication
ratioTR = 0.8 #Percentage for training
#create random 80/20 % split
trainIndex <- createDataPartition(fdata$Y,      #output variable. createDataPartition creates proportional partitions
                                  p = ratioTR,  #split probability
                                  list = FALSE, #Avoid output as a list
                                  times = 1)    #only one partition
#obtain training and validation sets
fTR = fdata[trainIndex,]
fTV = fdata[-trainIndex,]


## Initialize trainControl -----------------------------------------------------------------------
#  no cross-validation method is used
ctrl <- trainControl(method = "none",                      # method= "none" when no resampling is used
                     summaryFunction = defaultSummary,    #Performance summary for comparing models in hold-out samples
                     classProbs = TRUE,                    #Compute class probs in Hold-out samples
                     returnResamp = "all",                 #Return all information about resampling
                     savePredictions = TRUE)               #Compute class probs in Hold-out samples

ctrl_tune <- trainControl(method = "cv",                        #k-fold cross-validation
                          number = 10,                          #Number of folds
                          summaryFunction = defaultSummary,    #Performance summary for comparing models in hold-out samples
                          classProbs = TRUE,                    #Compute class probs in Hold-out samples
                          returnResamp = "all",                 #Return all information about resampling
                          savePredictions = TRUE)               #Compute class probs in Hold-out samples

#-------------------------------------------------------------------------------------------------
#----------------------------------- ORIGINAL TREE   ---------------------------------------------
#-------------------------------------------------------------------------------------------------

## FIT MODEL -------------------------------------------------------------------------------------------
set.seed(150) #For replication
#Train model using training data

ORIGINALtree.fit = train(fTR[,1:92], #Input variables. Other option: fdata[,1:2]
                         y = fTR$Y, 
                         method = "rpart",    
                         parms = list(split = "gini"),          # impuriry measure
                         preProcess = c("center","scale"),
                         tuneGrid = data.frame(cp = seq(0,0.4,0.05)),
                         trControl = ctrl_tune,    #ctrl_tune
                         metric = "ROC")

ORIGINALtree.fit #information about the resampling settings
summary(ORIGINALtree.fit)  #information about the model trained
ORIGINALtree.fit$finalModel #Cuts performed and nodes. Also shows the number and percentage of cases in each node.

#Plot the tree:
plot(ORIGINALtree.fit$finalModel, uniform=TRUE,margin=0.2)
text(ORIGINALtree.fit$finalModel, use.n=TRUE, all=TRUE, cex=0.8)

#Measure for variable importance
varImp(ORIGINALtree.fit,scale = FALSE)
plot(varImp(ORIGINALtree.fit,scale = FALSE))

tree.fit = ORIGINALtree.fit;

## Evaluate model --------------------------------------------------------------------------------
#Evaluate the model with training and validation sets
#training
fTR_eval = fTR
fTR_eval$tree_prob = predict(tree.fit, type="prob" , newdata = fTR) # predict probabilities
fTR_eval$tree_pred = predict(tree.fit, type="raw" , newdata = fTR) # predict classes 
#validation
fTV_eval = fTV
fTV_eval$tree_prob = predict(tree.fit, type="prob" , newdata = fTV) # predict probabilities
fTV_eval$tree_pred = predict(tree.fit, type="raw" , newdata = fTV) # predict classes 


## Performance measures: confusion matices
# Training
confusionMatrix(data = fTR_eval$tree_pred, #Predicted classes
                reference = fTR_eval$Y, #Real observations
                positive = "YES") #Class labeled as Positive
# Validation
confusionMatrix(fTV_eval$tree_pred,  fTV_eval$Y, positive = "YES")




#-------------------------------------------------------------------------------------------------
#----------------------------------- BAGGED TREE     ---------------------------------------------
#-------------------------------------------------------------------------------------------------

set.seed(150) #For replication
#Train model using boostrapped trained data
BAGtree.fit = train(fTR[,1:92], #Input variables. Other option: fdata[,1:2]
                    y = fTR$Y, 
                    method = "treebag",        #  <------------------------------        BAGGING TREE
                    preProcess = c("center","scale"),
                    trControl = ctrl, 
                    metric = "ROC")

#Plot one individual tree (the third one)
plot(BAGtree.fit$finalModel$mtrees[[3]]$btree, uniform=TRUE,margin=0.2)
text(BAGtree.fit$finalModel$mtrees[[3]]$btree, use.n=TRUE, all=TRUE, cex=.8)

tree.fit = BAGtree.fit;

## Evaluate model --------------------------------------------------------------------------------
#Evaluate the model with training and validation sets
#training
fTR_eval = fTR
fTR_eval$tree_prob = predict(tree.fit, type="prob" , newdata = fTR) # predict probabilities
fTR_eval$tree_pred = predict(tree.fit, type="raw" , newdata = fTR) # predict classes 
#validation
fTV_eval = fTV
fTV_eval$tree_prob = predict(tree.fit, type="prob" , newdata = fTV) # predict probabilities
fTV_eval$tree_pred = predict(tree.fit, type="raw" , newdata = fTV) # predict classes 


## Performance measureS: confusion matices
# Training
confusionMatrix(data = fTR_eval$tree_pred, reference = fTR_eval$Y, positive = "Summer")
# Validation
confusionMatrix(fTV_eval$tree_pred, fTV_eval$Y,  positive = "YES")


#-------------------------------------------------------------------------------------------------
#--------------------------------- RANDOM FORESTS    ---------------------------------------------
#-------------------------------------------------------------------------------------------------
#M ES EL NUMERO DE VARIABLES INDEPENDIENTES QUE COGE POR NODO PARA HACER LA CLASIFICACION
    #ASI ES COMO CONSIGUE QUE LAS VARIABLES NO ESTEN CORRELADAS

#m=33, t=500
set.seed(150) #For replication
#Training the model
RFtree.fit = train(fTR[,1:92], #Input variables. Other option: fdata[,1:2]
                   y = fTR$Y, # output variable
                   method = "rf", #Random forest
                   ntree = 1000,  #<---------------------------  Number of trees to grow
                   tuneGrid = data.frame(mtry = seq(1,ncol(fTR)-1)), # m parameter          
                   trControl = ctrl_tune, #Resampling settings 
                   metric = "ROC")    #Summary metrics
# See the forest
RFtree.fit

tree.fit = RFtree.fit;
#Evaluate the model with training and validation sets
#training
fTR_eval = fTR
fTR_eval$tree_prob = predict(tree.fit, type="prob" , newdata = fTR) # predict probabilities
fTR_eval$tree_pred = predict(tree.fit, type="raw" , newdata = fTR) # predict classes 
#validation
fTV_eval = fTV
fTV_eval$tree_prob = predict(tree.fit, type="prob" , newdata = fTV) # predict probabilities
fTV_eval$tree_pred = predict(tree.fit, type="raw" , newdata = fTV) # predict classes 

#Plot classification in a 2 dimensional space
Plot2DClass(fTR[,1:2], #Input variables of the model
            fTR$Y,     #Output variable
            tree.fit,#Fitted model with caret
            var1="X1", var2="X2", #variables that define x and y axis
            selClass = "YES")     #Class output to be analyzed 

## Performance measureS: confusion matices
# Training
confusionMatrix(data = fTR_eval$tree_pred, reference = fTR_eval$Y, positive = "YES")
# Validation
confusionMatrix(fTV_eval$tree_pred, fTV_eval$Y,  positive = "YES")


