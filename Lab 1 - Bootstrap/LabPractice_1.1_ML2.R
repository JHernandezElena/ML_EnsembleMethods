#################################################################################
##############     LabPractice 1.1 BOOTSTRAP         ############################
##############        MACHINE LEARNING II            ############################
##############           ESU   Jan-2020              ############################
#################################################################################

## Set working directory --------------------------------------------------------

## Load libraries ---------------------------------------------------------------
library(caret)
library(ggplot2)
library(GGally)
library(boot)


# remove all existing variables--------------------------------------------------
rm(list=ls())

# SAMPLE MEAN AND VARIANCE ESTIMATORS using BOOTSTRAP----------------------------------------------------------------
set.seed(150) #For replication
sampleSize = 10000;
y<- 10 + rnorm (sampleSize, mean=0.0, sd=2.0); # normal N(0,2)

hist(y,100);

# Estimate the mean and variance from the sample
meanSample = mean(y)
varSample = var(y)

#theory 
varSampleMean = varSample / sampleSize;


#  The Bootstrap approach
# One of the great advantages of the bootstrap approach is that it can be
# applied in almost all situations. No complicated mathematical calculations
# are required. Performing a bootstrap analysis in R entails only two steps.

# First, we must create a function that computes the statistic of interest.
# Second, we use the boot() function, which is part of the boot library, to
    # boot() perform the bootstrap by repeatedly sampling observations from the data
    # set with replacement.

bootMeanVar.fn=function (data ,index){
  m = mean(data[index])
  v = var(data[index])
  return (c(m,v))
}

#Check the function para ver que funciona
bootMeanVar.fn(y, 1:100)
bootMeanVar.fn(y, 1:1) # variance ??

# Produce R bootstrap estimates for the mean and the variance of y
bb = boot(y, bootMeanVar.fn,R=10000) #le pasamos el dataset, la funcion de lo que queramos calcular y el numero de samples que queremos

hist(bb$t[,1],40) #nos devuelve las medias (lo primero que hay en la fn) de las 10000 muestras
hist(bb$t[,2],40) #nos devuelve las varianzas(lo segundo de la fn) de las 100000 muestras

# Empirical joint distribution of estimators
coefsdf = data.frame(bb$t)
colnames(coefsdf)<- c("Mean","Var")
ggpairs(coefsdf)

# compute MSE
mean((y - meanSample)^2)


# MIN AND MAX ESTIMATION USING BOOTSTRAP --------------------------------------------------------

# dataset
set.seed(150) #For replication
sampleSize = 5000;
y<- 10 + rnorm (sampleSize, mean=0.0, sd=2.0); # normal N(0,2) 
  #probar con runif(sampleSize, 0, 10) para ver la diferencia
  #la distribucion del min estara mucho mas pegada al 0 ya que hay mas probabilidad de encontrarnos valores extremos

# estimates from the sample
minSample = min(y)
maxSample = max(y)
rangeSample = maxSample - minSample

bootMinMax.fn=function (data ,index){
  mm = min(data[index])
  MM = max(data[index])
  difMMmm = MM - mm
  return (c(mm,MM, difMMmm))
}

#Check the function
bootMinMax.fn(y, 1:100)
bootMinMax.fn(y, 1:10)

# Produce R bootstrap estimates for the mean and the variance of y
bb = boot(y, bootMinMax.fn,R=1000)

hist(bb$t[,1],40, main = 'Mininum distribution') 
  #tiene sentido ya que cada valor del histograma es si se coge el valor mas pequeno de la muestra, si no el siguiente mas pequeno, etc debido al reemplazamiento
hist(bb$t[,2],40, main = 'Maximum distribution')
hist(bb$t[,3],40, main = 'Range distribution')

# Empirical joint distribution of estimators
coefsdf = data.frame(bb$t)
colnames(coefsdf)<- c("Min","Max", "Range")
ggpairs(coefsdf)

minSampleBB = mean(bb$t[,1])
maxSampleBB = mean(bb$t[,2])
rangeSampleBB = mean(bb$t[,3])

# ------------------------------------------------------------------------
# LINEAR REGRESSION ESTIMATION -------------------------------------------
# ------------------------------------------------------------------------
set.seed(150) #For replication
sampleSize = 100;

# sample
x <-runif(sampleSize,0,1);
y <- 1  + 5 * x + rnorm (sampleSize, mean=0, sd=3.0);

SSample = data.frame(x,y)
ggpairs(SSample)

bootBetasLM.fn=function (data ,index){
  return (coef(lm("y~x", data=data[index,])))  #DEVUELVE LOS COEFICIENTES DEL AJUSTE DEL MODELO
}

#Check the function
bootBetasLM.fn(SSample, 1:10)
summary(lm("y~x", data=SSample ,subset =1:10))$coefficients

# Produce R = 10000 bootstrap estimates for beta0 and beta1
bb = boot(SSample, bootBetasLM.fn,R=10000)


#original (using the full sample)-EL MODELO QUE OBTENDRIAMOS ESTIMANDO NORMAL CON LA MUESTRA ENTERA
coef_original <- coef(lm("y~x", data=SSample))
x = lm("y~x", data=SSample)

#bias and std.error
# mean model: mean(bb$t[,1])
biasBeta0 <- coef_original[1] - mean(bb$t[,1])
biasBeta1 <- coef_original[2] - mean(bb$t[,2])
stderrBeta0 <- sd(bb$t[,1])
stderrBeta1 <- sd(bb$t[,2])


# Empirical joint distribution of beta0 and beta1
plot(bb$t[,2], bb$t[,1]) #CADA PUNTO ES UN MODELO (una linea ajustada por B0, B1)

coefsdf = data.frame(bb$t)
colnames(coefsdf)<- c("Beta0","Beta1")
ggpairs(coefsdf)  #COMO PODEMOS COMPROBAR LAS CAMPANAS ESTAN CENTRADAS EN LO QUE OBTENEMOS EN COEF_ORIGINAL


###############################################
##Obtener el estimate del error con bootstrap##
###############################################
bootRmseLM.fn=function (data ,index){
  rr = residuals(lm("y~x", data=data[index,]))
  rmse = sqrt(mean(rr^2))
  return (rmse)  
}

#Check the function (COMPROBAMOS QUE OBTENEMOS EL MISMO RMSE o parecido)
bootRmseLM.fn(SSample, 1:100)
summary(lm("y~x", data=SSample ,subset =1:100))

# Produce R = 10000 bootstrap estimates for beta0 and beta1
bb = boot(SSample, bootRmseLM.fn,R=10000)

hist(bb$t, 40, main ='rmse distribution')


#original (using the full sample)-EL RMSE QUE OBTENDRIAMOS ESTIMANDO NORMAL CON LA MUESTRA ENTERA
summary(lm("y~x", data=SSample))



##HACER CON lm(y~x^2) o lm(y~x+x^2) **mirar cual de las dos formas es
    #METER EL MODELO DE SOLO UNA CONSTANTE QUE NO DEPENDA DE LA X
    #comparar los histogramas y sacar el 25 y el 75 percentil
    
#hacer grafica de eje y = rmse, eje x = las distribuciones con orden 1, orden 2, etc...