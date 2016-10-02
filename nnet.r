#No correr con menos de 12 gib de memoria. Para probar con menos bajar la cantidad de elementos

library(NLP)
library(tm)
library(nnet)

corpus <- Corpus(VectorSource(trainPreProcesado$Text))
dtm <- DocumentTermMatrix(corpus)
#Con 0.99 quedan mas o menos 650 terminos. Con mas empiezo a tener problemas de memoria
dtmr <- removeSparseTerms(dtm, 0.99)
rm(corpus)
rm(dtm)
m <- as.matrix(dtmr)
#Normalizo
#Scale no se comporta bien para matrices grandes
#m <- scale(m, center = FALSE, scale = TRUE)
#Apply es como map, y sweep "hereda" de apply
#2 indica que por columnas aplico la funcion maximo.
maxs <- apply(m,2, max)
#Con sweep divido cada columna por el maximo
#Cuidado que a veces no lo toma y hay que mandarlo de nuevo
m <- sweep(m, 2, maxs, FUN="/")
rm(maxs)
#Con 250000 datos me alcanzan mis 12 gib de memoria ram. Con mas no, pero deberia poder variar.
m <- m[1:250000,]
trainReducido <- trainPreProcesado[1:250000,]
nn <- nnet(m[1:250000], (trainPreProcesado$Prediction-1)/4, size = 1)

#Para testear, y liberar algo de la memoria antes. Si el sistema operativo se queda sin nada
#usar un gc()
m <- as.matrix(dtmr)
m <- sweep(m, 2, maxs, FUN="/")
m<-m[300000:400000,]
rm(maxs)
prediction <- predict(nn, m)
prediction <- (prediction+0.25)*4
real <- trainPreProcesado$Prediction[300000:400000]
diferencia <- prediction - realM
diferenciaCuadrada <- diferencia^2
sumaDiferenciasCuadradas <- colSums(diferenciaCuadrada)
varianza <- sumaDiferenciasCuadradas/nrow(diferenciaCuadrada)
desvio <- sqrt(varianza)
