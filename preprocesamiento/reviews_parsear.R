library("tm")
reviews <- read.csv("~/Downloads/train.csv")

stopwords(kind="en")
procesarTexto <- function(texto) {
  stripWhitespace(removePunctuation(removeWords(tolower(texto), stopwords(kind="en"))))
}

procesarVectorTexto <- Vectorize(procesarTexto)

reviews$Text <- procesarVectorTexto(reviews$Text)

write.csv(reviews,"~/Downloads/train_processed.csv")
