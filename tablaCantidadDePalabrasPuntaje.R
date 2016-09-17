library("NLP")
library("SnowballC")
library("tm")
library("slam")
#5 o cualquier puntaje que se quiera
train1 <- trainPreProcesado[trainPreProcesado$Prediction == 5,]
corpus <- Corpus(VectorSource(train1[, "Text"]))
rm(train1)
tdm <- DocumentTermMatrix(corpus)
rm(corpus)
#Esto deja una matriz que puede estar hasta un 99% dispersa(?). No estoy seguro de la medida
#Pero queda menos dispersa.
#En consecuencia todo lo que queda son palabras utiles. Si se quieren mas
#Se sube a 99.9 por ejemplo
tdms <- removeSparseTerms(tdm, 0.99)
rm(tdm)
freq <- col_sums(tdms)
ord <- order(freq)
#Para que no se sobre escriban las tablas ir variando el nombre 
#table5, por otros, como por ejemplo table1, table2.
table5 = freq[ord]
rm(tdms)
rm(freq)
rm(ord)