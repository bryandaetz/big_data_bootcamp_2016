library(jsonlite)
library(dplyr)
library(magrittr)
library(tm)
library(topicmodels)

set.seed(42)


# Load data
sotu <- fromJSON(txt = "C://Users/bdaet/Desktop/myProjects/big_data_bootcamp_2016-master/sotu_parsed.json")


#adding extra stopwords (common words in the English language that won't provide us with much information)
stopwords_extra <- c(stopwords(kind = 'en'), 'will', 'thank', 'can')

#creating a Document Term Matrix
dtm <- DocumentTermMatrix(x = Corpus(VectorSource(sotu$content)),
                          control = list(tokenize = words,        #tokenizing text
                                         tolower = TRUE,          #making text lower case
                                         stopwords = stopwords_extra,   #applying added stopwords
                                         stemming = 'english',          #converting words to their stems     
                                         removePunctuation = TRUE,      #removing punctuation
                                         removeNumbers = TRUE,          #removing numbers
                                         minDocFreq = length(sotu$content) * 0.05,
                                         maxDocFreq = length(sotu$content) * 0.80,
                                         weighting = weightTf))
#creating a LDA (latin dirichlet allocation) model
lda <- LDA(x = dtm, k = 5, method = 'VEM')
terms(x = lda, k = 10)
topics(x = lda, k = 5)

# DF: Terms
term_extract <- terms(lda, k = 25)
v_topics <- c(); v_terms <- c(); v_ranks <- c(); v_betas <- c()
for (topic in 1:lda@k) {
  for (term in 1:25) {
    v_topics <- c(v_topics, topic)
    v_terms <- c(v_terms, term_extract[term, topic])
    v_ranks <- c(v_ranks, term)
    v_betas <- c(v_betas, lda@beta[topic, which(lda@terms == term_extract[term, topic])])
  }
}
term_rankings <- data_frame(topic = v_topics,
                            term = v_terms,
                            rank = v_ranks,
                            term_beta = v_betas)
rm(v_topics, v_terms, v_ranks, v_betas)

# DF: Document Distributions
v_topics <- c(); v_docs <- c(); v_gammas <-c()
for (topic in 1:lda@k) {
  for (doc in 1:length(lda@documents)) {
    v_topics <- c(v_topics, topic)
    v_docs <- c(v_docs, lda@documents[doc])
    v_gammas <- c(v_gammas, lda@gamma[doc, topic])
  }
}
doc_distributions <- data_frame(topic = v_topics,
                                doc = v_docs,
                                doc_gamma = v_gammas)
rm(v_topics, v_docs, v_gammas)