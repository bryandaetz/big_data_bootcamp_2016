library(dplyr)
library(magrittr)
library(tm)
library(topicmodels)
library(jsonlite)


set.seed(1300)
options(mc.cores = 1)


sotu = fromJSON(txt = './sotu_parsed.json')

# TM Pipeline
sotu_corpus <- Corpus(VectorSource(sotu$content)) %>%
    tm_map(x = ., FUN = PlainTextDocument) %>%
    tm_map(x = ., FUN = removePunctuation) %>%
    tm_map(x = ., FUN = removeNumbers) %>%
    tm_map(x = ., FUN = removeWords, stopwords(kind = 'en')) %>%
    tm_map(x = ., FUN = stripWhitespace)



stopwords_extra <- c(stopwords(kind = 'en'), 'will', 'thank', 'can')
dtm <- DocumentTermMatrix(x = Corpus(VectorSource(sotu$content)),
                          control = list(tokenize = words,
                                         tolower = TRUE,
                                         stopwords = stopwords_extra,
                                         stemming = TRUE,
                                         removePunctuation = TRUE,
                                         removeNumbers = TRUE,
                                         minDocFreq = length(sotu$content) * 0.05,
                                         maxDocFreq = length(sotu$content) * 0.80,
                                         weighting = weightTf))


doc_term <- DocumentTermMatrix(sotu_corpus)
doc_term$dimnames$Docs <- sotu$file_name

tf_idf <- weightTfIdf(m = doc_term, normalize = TRUE)
tf_idf_mat <- as.matrix(tf_idf)
tf_idf_dist <- dist(tf_idf_mat, method = 'euclidean')


# Cluster: KMeans
tf_idf_norm <- tf_idf_mat / apply(tf_idf_mat, MARGIN = 1, FUN = function(x) sum(x^2)^0.5)
km_clust <- kmeans(x = tf_idf_norm, centers = 5, iter.max = 25)
pca_comp <- prcomp(tf_idf_norm)
pca_rep <- data_frame(sotu_name = sotu$file_name,
                      pc1 = pca_comp$x[,1],
                      pc2 = pca_comp$x[,2],
                      clust_id = as.factor(km_clust$cluster))


# LDA
lda <- LDA(x = dtm, k = 8, method = 'VEM')

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

























