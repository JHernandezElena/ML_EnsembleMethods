#################################################################################
##############      LabPractice 1.4 BOOSTING         ############################
##############        MACHINE LEARNING II            ############################
##############           ESU   Jan-2020              ############################
#################################################################################

## Set working directory --------------------------------------------------------
getwd()

## Load libraries ---------------------------------------------------------------
library(caret)
library(ggplot2)
library(ROCR) #for plotting ROC curves.
library(MLTools)

library(gbm)
#library(plyr)

## remove all existing variables--------------------------------------------------
rm(list=ls())

## Load file -------------------------------------------------------------------------------------
fdata = read.table("SimData.dat", sep = "", header = TRUE, stringsAsFactors = FALSE)
str(fdata)
fdata$Y = as.factor(fdata$Y)
str(fdata)

## Exploratory analysis --------------------------------------------------------------------------------
ggplot(fdata)+geom_point(aes(x=X1,y=X2,color=Y))

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
                     summaryFunction = twoClassSummary,    #Performance summary for comparing models in hold-out samples
                     classProbs = TRUE,                    #Compute class probs in Hold-out samples
                     returnResamp = "all",                 #Return all information about resampling
                     savePredictions = TRUE)               #Compute class probs in Hold-out samples

ctrl_tune <- trainControl(method = "cv",                        #k-fold cross-validation
                          number = 10,                          #Number of folds
                          summaryFunction = twoClassSummary,    #Performance summary for comparing models in hold-out samples
                          classProbs = TRUE,                    #Compute class probs in Hold-out samples
                          returnResamp = "all",                 #Return all information about resampling
                          savePredictions = TRUE)               #Compute class probs in Hold-out samples

#-------------------------------------------------------------------------------------------------
#----------------------------------- ORIGINAL TREE   ---------------------------------------------
#-------------------------------------------------------------------------------------------------

## FIT MODEL -------------------------------------------------------------------------------------------
set.seed(150) #For replication
#Train model using training data
ORIGINALtree.fit = train(fTR[,c("X1","X2")], #Input variables. Other option: fdata[,1:2]
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

#Plot classification in a 2 dimensional space
Plot2DClass(fTR[,1:2], #Input variables of the model
            fTR$Y,     #Output variable
            tree.fit,#Fitted model with caret
            var1="X1", var2="X2", #variables that define x and y axis
            selClass = "YES")     #Class output to be analyzed 

## Performance measures: confusion matices
# Training
confusionMatrix(data = fTR_eval$tree_pred, #Predicted classes
                reference = fTR_eval$Y, #Real observations
                positive = "YES") #Class labeled as Positive
# Validation
confusionMatrix(fTV_eval$tree_pred,  fTV_eval$Y, positive = "YES")




#-------------------------------------------------------------------------------------------------
#----------------------------------- GRADIENT BOOSTING -------------------------------------------
#-------------------------------------------------------------------------------------------------

# Use Stochastic Gradient Boosting (gbm)

# Prepare the tune grid
#     - number of trees (n.trees), 
#     - depth of trees (interaction.depth)
#     - shrinkage (shrinkage), 
#     - Min. Terminal Node Size (n.minobsinnode) 

gbmGrid <- expand.grid(.n.trees = seq(100, 1000, by = 50), #tampoco se pueden poner mazo arboles
                       .interaction.depth = seq(1, 10, by = 1), #queremos que aprenda DESPACIO
                       .shrinkage = c(0.01, 0.1), # two typical values
                       .n.minobsinnode = c(1))

# Train the GBM
set.seed(150)
GBM.fit =  train(fTR[,c("X1","X2")], #Input variables. Other option: fdata[,1:2]
                 y = fTR$Y,
                 method = "gbm",
                 tuneGrid = gbmGrid,
                 trControl = ctrl_tune,    #use cv
                 metric = "ROC",
                 verbose = FALSE) # The gbm() function produces copious amounts of output, avoid printing a lot

GBM.fit #information about the resampling settings

plot(GBM.fit)

#Measure for variable importance
varImp(GBM.fit,scale = FALSE)
plot(varImp(GBM.fit,scale = FALSE))

#Plot OOB
plot(GBM.fit$finalModel$oobag.improve) # shows the oobag improvement evolving with n.trees

tree.fit = GBM.fit;
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

#Plot classification in a 2 dimensional space
Plot2DClass(fTR[,1:2], #Input variables of the model
            fTR$Y,     #Output variable
            tree.fit,#Fitted model with caret
            var1="X1", var2="X2", #variables that define x and y axis
            selClass = "YES")     #Class output to be analyzed 

## Performance measures: confusion matices
confusionMatrix(fTR_eval$tree_pred, fTR_eval$Y, positive = "YES") # Training
confusionMatrix(fTV_eval$tree_pred,  fTV_eval$Y, positive = "YES")# Validation
