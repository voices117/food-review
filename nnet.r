library(NLP)
library(tm)
library(nnet)
library(dplyr)
#Trabajo con pocos datos para testear, dummy test
set.seed(123)
#Tiro datos ya desde un inicio, esto es para acelerar pruebas. 
#Se puede poner en 1 o comentar para no tirar

preProcesado<-sample_frac(trainPreProcesado, 0.5)

corpus <- Corpus(VectorSource(preProcesado$Text))
#Actualmente usa TF, puede cambiarse por TF-IDF
dtm <- DocumentTermMatrix(corpus, control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE)))
#Con 0.99 quedan mas o menos 650 terminos. Con mas empiezo a tener problemas de memoria
rm(corpus)
dtmr <- removeSparseTerms(dtm, 0.99)
rm(dtm)
m <- as.matrix(dtmr)
#Normalizo
  m <- scale(m)
  
#Preparo el train, va a ser chico para que alcance la memoria y el tiempo
set.seed(123)
filasTrain <- sample(nrow(m), size = floor(nrow(m)*0.5))
train <- m[filasTrain,]

#Armo el vector Y para ajustar
trainY <- preProcesado[filasTrain,]$Prediction
#Nota: La Y no es necesario normalizarla en una red Neuronal. Asi lo lei en varios sitios
nn <- nnet(train, trainY, size = 1, linout = TRUE)

#Libero la memoria del train, y me quedo con el test
rm(train)
test <- m[-filasTrain,]
#Me quedo con el test
prediction <- predict(nn, test)
head(prediction)