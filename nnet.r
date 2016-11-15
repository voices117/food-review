library(NLP)
library(tm)
library(nnet)
library(SnowballC)
library(dplyr)
library(slam)

ptmInicial <- proc.time()

modeloLineal = TRUE;
#454760 es el tamanio del set original de train
MAX_DATA <- 454760
#testResults <- matrix(nrow = 8, ncol = 3)

load(file = "trainTest.rdata")
trainTestText <- trainTest$Text
rm(trainTest)
corpus <- Corpus(VectorSource(trainTestText))
rm(trainTestText)
#Actualmente usa TF, puede cambiarse por TF-IDF
dtm <- DocumentTermMatrix(corpus, control = list(weighting = function(x) weightTf(x)))

#Con 0.99 quedan mas o menos 650 terminos. Con mas empiezo a tener problemas de memoria
rm(corpus)
dtmr <- removeSparseTerms(dtm, 0.99)
rm(dtm)

#Busco los datos que necesito para estandarizar la matriz
media <- col_means(dtmr)
varianza <- colapply_simple_triplet_matrix(dtmr, FUN = var)
desvio <- sqrt(varianza)
m <- as.matrix(dtmr)
#scale(m)
m[1:ncol(m),] <- m[1:ncol(m),] - media[1:ncol(m)]
m[1:ncol(m),] <- m[1:ncol(m),] / desvio[1:ncol(m)]

dtmr <- as.simple_triplet_matrix(m)

#Me quedo unicamente con lo que voy a usar para entrenar y testear
rm(media,varianza,desvio)
save(dtmr, file = "dtmrTemp.RData")
rm(dtmr)

#Saco los datos del train de kaggle de la memoria
m <- m[1:MAX_DATA,]

#Preparo el train
set.seed(345)
filasTrain <- sample(nrow(m), size = floor(nrow(m)*0.9))
train <- m[filasTrain,]
#Dejo unicamente el train, el test lo puedo recuperar mas adelante
rm(m)
#Armo el vector Y para ajustar
trainOriginal <- read.csv(file = "train.csv")
trainY <- trainOriginal[filasTrain,]$Prediction
rm(trainOriginal)

ptmFinal <- proc.time()

if(modeloLineal){
  nn <- nnet(train, trainY, size = 1, linout = TRUE, maxit = i*100)
}else{
  trainYb <- matrix(nrow = length(trainY), ncol = 5)
  for(i in 1:5){
    trainYb[,i] <- ifelse(trainY == i,1,0)
  }
  trainY <- trainYb
  rm(trainYb)
  nn <- nnet(train, trainY, size = 1, linout = FALSE, MaxNWts = 10000, skip = T)
}

#rm(train)
test <- m[-filasTrain,]
#Me quedo con el test
prediction <- predict(nn, test)

if(modeloLineal){
  acertados <- sum(round(prediction) == preProcesado[-filasTrain,]$Prediction)
  total <- length(preProcesado[-filasTrain,]$Prediction)
  cat("Test: ",(acertados/total)*100, "%")
  
  prediction <- predict(nn, train)
  acertados <- sum(round(prediction) == preProcesado[filasTrain,]$Prediction)
  total <- length(preProcesado[filasTrain,]$Prediction)
  cat("Train:: ",(acertados/total)*100, "%")
  testResults[i,] <- cbind(acertados,total, acertados/total)
  write.csv(testResults, file = "testResults.csv")
}

