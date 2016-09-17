library(NLP)
library(tm) #si nunca la instalaron usar install.packages("tm")
library(SnowballC) # idem install.packages("SnowballC")

train = read.csv("train.csv")
myCorpus <- Corpus(VectorSource(train$Text)) 
## Preprocessing    
myCorpus <- tm_map(myCorpus, removePunctuation, mc.cores = 1)   # *Removing punctuation:*  
#myCorpus <- tm_map(myCorpus, removeNumbers)      # *Removing numbers:*  
myCorpus <- tm_map(myCorpus, content_transformer(tolower), mc.cores=1)   # *Converting to lowercase:*  
myCorpus <- tm_map(myCorpus, removeWords, stopwords("english"), mc.cores = 1)   # *Removing "stopwords"
myCorpus <- tm_map(myCorpus, stemDocument, mc.cores = 1)   # *Removing common word endings* (e.g., "ing", "es") 
myCorpus <- tm_map(myCorpus, stripWhitespace, mc.cores = 1)   # *Stripping whitespace  

#A partir de aca guardo los resultados en un dataframe igual al original, reemplazando
#el campo Text por el text pre procesado
trainPreProcesado = train
dataframe<-data.frame(text=unlist(sapply(myCorpus, `[`, "content")))
trainPreProcesado[,"Text"] <- dataframe[,"text"]
write.csv(trainPreProcesado, file = "trainPreProcesado.csv")
