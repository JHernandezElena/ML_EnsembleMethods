#################################################################################
##############     LabPractice 1.2 BAGGING           ############################
##############        MACHINE LEARNING II            ############################
##############           ESU   Jan-2019              ############################
#################################################################################

## Set working directory --------------------------------------------------------
getwd()

## Load libraries ---------------------------------------------------------------
library(caret)
library(ggplot2)
library(ROCR) #for plotting ROC curves.
library(MLTools)

## remove all epxisting variables--------------------------------------------------
rm(list=ls())

## Load functions --------------------------------------------------------------------------------
#Source plot2Dclass function, a custom function useful for plotting the results of classification models 
#with two input variables and one output.
#source("plot2Dclass.R")


## Load file -------------------------------------------------------------------------------------
fdata = read.table("SimData.dat", sep = "", header = TRUE, stringsAsFactors = FALSE)
str(fdata)
fdata$Y = as.factor(fdata$Y)
str(fdata)

## Exploratory analysis -------------------------------------------------------------------------------------
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
#first, with fixed parameters
#Train model using training data

ORIGINALtree.fit = train(fTR[,c("X1","X2")], #Input variables. Other option: fdata[,1:2]
                         y = fTR$Y, 
                         method = "rpart", 
                         preProcess = c("center","scale"),
                         tuneGrid = data.frame(cp = 0.001),  # 0: very complex  
                         trControl = ctrl,    #ctrl_tune
                         metric = "ROC")

            #Cp = complexity param = 0.01 -> necesitamos mejorar un 1% la pureza del siguiente nodo para realizar un nuevo corte
            #The GRANDE the CP MAS SIMPLE EL MODELO


ORIGINALtree.fit = train(fTR[,c("X1","X2")], #Input variables. Other option: fdata[,1:2]
                 y = fTR$Y, 
                 method = "rpart",    
                 parms = list(split = "gini"),          # impuriry measure
                 preProcess = c("center","scale"),
                 tuneGrid = data.frame(cp = seq(0,0.4,0.05)),
                 trControl = ctrl_tune,    #ctrl_tune
                 metric = "ROC") 

    #MAXIMIZA EL AREA BAJO LA CURVA CON LOS 10 K-FOLDS DE CV PARA ELEGIR EL CP OPTIMO Y LUEGO ENTRENA CON TODOS LOS DATOS DE TRAIN
    #prueba cada valor del hyper parametro para 9 folds dejando 1 out con el que calcular el error, luego hace la media de los 10
          #prueba el siguiente valor cp para los folds.. etc

ORIGINALtree.fit #information about the resampling settings

ggplot(ORIGINALtree.fit)

summary(ORIGINALtree.fit)  #information about the model trained
ORIGINALtree.fit$finalModel #Cuts performed and nodes. Also shows the number and percentage of cases in each node.

#Plot the tree:
plot(ORIGINALtree.fit$finalModel, uniform=TRUE,margin=0.2)
text(ORIGINALtree.fit$finalModel, use.n=TRUE, all=TRUE, cex=0.6)
      #En los nodos terminales de dice la clase mayoritaria y cuantos casos(ej: YES 10/184 -> hay 184 sies y 10 noes)
      #el NO 400/400 es la cantidad de datos (400 nos 400 sies) que hay antes de partir
              #pone No porque la N va antes de la Y

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


## Performance measures --------------------------------------------------------------------------------

#######confusion matices
    #MIRAR EL 95% CI QUE DICE CUANTO CAMBIA AL 95% DE CONF EL ACCURACY
    #SI EL ERROR EN VALIDACION SALE MAS BAJO QUE EN ENTRENAMIENTO TENEMOS PROBLEMAS:
        #significa que el conjunto de test es mas facil
        #el reparto 80/20 no es valido
        #hay que volver atras y cambiar la seed de la division

# Training
confusionMatrix(data = fTR_eval$tree_pred, #Predicted classes
                reference = fTR_eval$Y, #Real observations
                positive = "YES") #Class labeled as Positive
# Validation
confusionMatrix(fTV_eval$tree_pred, 
                fTV_eval$Y, 
                positive = "YES")

# #######Classification performance plots 
# # Training
PlotClassPerformance(fTR_eval$Y,       #Real observations
                    fTR_eval$tree_prob,  #predicted probabilities
                    selClass = "YES") #Class to be analyzed
# # Validation
PlotClassPerformance(fTV_eval$Y,       #Real observations
                     fTV_eval$tree_prob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed)




#-------------------------------------------------------------------------------------------------
#----------------------------------- BAGGED TREE     ---------------------------------------------
#-------------------------------------------------------------------------------------------------

set.seed(150) #For replication
#Train model using boostrapped trained data
BAGtree.fit = train(fTR[,c("X1","X2")], #Input variables. Other option: fdata[,1:2]
                 y = fTR$Y, 
                 method = "treebag",        #  <------------------------------        BAGGING TREE
                 preProcess = c("center","scale"),
                 trControl = ctrl, 
                 metric = "ROC")

#Plot one individual tree (the third one)
plot(BAGtree.fit$finalModel$mtrees[[3]]$btree, uniform=TRUE,margin=0.2)
text(BAGtree.fit$finalModel$mtrees[[3]]$btree, use.n=TRUE, all=TRUE, cex=.5)
      #VEMOS QUE SON MUY PROFUNDIS
      #SON ARBOLES QUE SE APRENDEN EL RUIDO 
      #MUCHO SESGO Y POCA VARIANZA

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

#Plot classification in a 2 dimensional space
Plot2DClass(fTR[,1:2], #Input variables of the model
            fTR$Y,     #Output variable
            tree.fit,#Fitted model with caret
            var1="X1", var2="X2", #variables that define x and y axis
            selClass = "YES")     #Class output to be analyzed 


## Performance measures --------------------------------------------------------------------------------

#######confusion matices
# Training
confusionMatrix(data = fTR_eval$tree_pred, #Predicted classes
                reference = fTR_eval$Y, #Real observations
                positive = "YES") #Class labeled as Positive
    #se aprende mazo los datos

# Validation
confusionMatrix(fTV_eval$tree_pred, fTV_eval$Y,  positive = "YES")
    #peores resultados
    #hay sobre-entrenamiento y luego no le va tan bien

# 
# #######Classification performance plots 
# # Training
# PlotClassPerformance(fTR_eval$Y,       #Real observations
#                      fTR_eval$tree_prob,  #predicted probabilities
#                      selClass = "YES") #Class to be analyzed
# # Validation
# PlotClassPerformance(fTV_eval$Y,       #Real observations
#                      fTV_eval$tree_prob,  #predicted probabilities
#                      selClass = "YES") #Class to be analyzed)



#EL MODELO ORIGINAL ES MEJOR!