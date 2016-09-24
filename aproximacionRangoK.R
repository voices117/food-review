#Reducir dimensiones, ejemplo con el de la guia

#Armo la matriz
c1 <- c(3,2,3,0,2)
c2 <- c(1,1,3,1,0)
c3 <- c(1,0,0,2,2)
c4 <- c(0,2,1,0,2)

m <- as.matrix(c1)
m <- cbind(m, c1)
m <- cbind(m, c2)
m <- cbind(m, c3)
m <- cbind(m, c4)
dimnames(m) <- NULL

#Calculado la dvs, con K columnas de V y U
K = 3
descomposicion <- svd(m,K,K)

#Armo la diagonal 
primerosDosAutovalores <- descomposicion$d[1:K]
diagonal <- diag(primerosDosAutovalores)

#Calculo la matriz aproximada de rango K , t transpone la matriz
mAproximacionDeRangoK = descomposicion$u %*% diagonal %*% t(descomposicion$v)