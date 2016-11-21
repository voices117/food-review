#Salida

MAX_DATA <- 454760
m <- as.matrix(dtmr)
m <- m[(MAX_DATA+1):nrow(m),]
prediction <- predict(nn, m)
testKaggle <- read.csv(file = "test.csv")
entregaKaggle <- cbind(testKaggle$Id, format(prediction, 
                                             scientific = FALSE, nsmall = 0))
colnames(entregaKaggle) <- c("Id","Prediction")
write.csv(entregaKaggle, file = "entregaKaggle.csv", row.names = FALSE)