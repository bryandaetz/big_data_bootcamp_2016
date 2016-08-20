import json, re

import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
import lda




def clean_tokenize(s, stop_words):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(s.lower())
    lemma = WordNetLemmatizer()
    clean = [lemma.lemmatize(token) for token in tokens if
        len(token) > 2 and
        not re.search('^\d+$', token) and  # scrub numbers if whole token
        token not in stop_words
    ]
    return clean

def load_clean_sotu(file_name):
    # Load Data
    with open(file_name, 'r') as f:
        sotu = json.loads(f.read())

    # Clean Documents
    sw = stopwords.words('english')
    for i in sotu:
        i['tokens'] = clean_tokenize(s = i['content'], stop_words = sw)

    # ['tokens', u'head2', u'char_content', u'file_name',
    # u'content', u'head1', u'year']

    return sotu

def build_cluster_model(data):
    doc_text = [' '.join(i['tokens']) for i in data]

    # TFIDF
    t_vec = TfidfVectorizer(
        analyzer = 'word',
        ngram_range = (1,1),
        use_idf = True,
        max_df = 0.8,
        min_df = 0.1
    )
    tfidf = t_vec.fit_transform(doc_text).toarray().swapaxes(0, 1)
    terms = t_vec.get_feature_names()

    # Get top n terms
    n_terms = len(terms)
    term_worth = tfidf.sum(1)
    s_index = np.argsort(term_worth)
    term_order = (n_terms - 1) - np.arange(n_terms).take(s_index.argsort())
    term_order = list(term_order[:100])

    terms = [terms[i] for i in term_order]
    tfidf = tfidf.take(term_order, axis = 0)
    mds = MDS(
        n_components = 2,
        max_iter = 100,
        random_state = 1300,
        dissimilarity = 'euclidean',
        n_jobs = 1,
        verbose = 0,
        eps = 1e-3,
        n_init = 3
    )
    points = mds.fit(tfidf).embedding_

    # Collect Points
    collector = {}
    for (idx, item) in enumerate(terms):
        collector[item] = {
            'x': (points[idx][0] - points[:,0].min()) / points[:,0].ptp(),
            'y': (points[idx][1] - points[:,1].min()) / points[:,1].ptp()
        }

    return (terms, collector, tfidf)

def cluster_points(points, k):
    km = KMeans(
        init = 'k-means++',
        n_clusters = k,
        n_init = 10,
        random_state = 1300
    )
    km.fit(points)
    km_labels = km.labels_
    return km_labels

def cluster_pipeline(data, k):
    terms, points, tfidf = build_cluster_model(data)

    collector = []
    x_pos = [points[i]['x'] for i in terms]
    y_pos = [points[i]['y'] for i in terms]
    doc_ids = [i['file_name'] for i in data]

    for (idx, item) in enumerate(doc_ids):
        entity = {}
        entity['id'] = item
        vec_weights = tfidf[:, idx]

        if np.sum(vec_weights) == 0.0:
            entity['x'] = 0.0
            entity['y'] = 0.0
        else:
            entity['x'] = np.average(a = x_pos, weights = vec_weights)
            entity['y'] = np.average(a = y_pos, weights = vec_weights)

        collector.append(entity)


    coll_points = np.array([[i['x'], i['y']] for i in collector])
    km_labels = cluster_points(coll_points, 3)

    for (idx, item) in enumerate(collector):
        collector[idx]['cluster_id'] = int(km_labels[idx])


    return collector

def build_lda_model(doc_text, doc_ids, lda_topics, max_df = 0.5, min_df = 0.05):
    # Build document vectors
    vec = CountVectorizer(
        analyzer = 'word',
        ngram_range = (1, 3),
        max_df = max_df,
        min_df = min_df
    )
    dtm = vec.fit_transform(doc_text)
    terms = vec.get_feature_names()
    n_terms = len(terms)
    n_docs = len(doc_ids)

    # Build LDA Model
    lda_model = lda.LDA(
        n_topics = lda_topics,
        n_iter = 2500,
        alpha = 0.1,
        eta = 0.01,
        random_state = 1300,
        refresh = 100
    )
    lda_model.fit(dtm)

    # Build Output Object
    output = {}
    output['num_topics'] = lda_topics
    output['log_likelihood'] = lda_model.loglikelihood()

    output['terms'] = []
    for (topic_id, topic_dist) in enumerate(lda_model.topic_word_):
        s_index = np.argsort(topic_dist)
        term_order = n_terms - np.arange(n_terms).take(s_index.argsort())
        for n in range(n_terms):
            output['terms'].append({
                'topic_id': topic_id,
                'term': terms[n],
                'rank': term_order[n],
                'beta': topic_dist[n]
            })

    output['docs'] = []
    for (topic_id, doc_dist) in enumerate(lda_model.doc_topic_.swapaxes(0,1)):
        for n in range(n_docs):
            output['docs'].append({
                'topic_id': topic_id,
                'doc_id': doc_ids[n],
                'gamma': doc_dist[n]
            })

    return output

def lda_pipeline(data, lda_topics):
    sotu_ids = [i['file_name'] for i in data]
    sotu_text = [' '.join(i['tokens']) for i in data]

    mod = build_lda_model(
        doc_text = sotu_text,
        doc_ids = sotu_ids,
        lda_topics = lda_topics
    )

    out_file = './lda_models/lda_k{k}.json'.format(k = lda_topics)
    with open(out_file, 'w') as f:
        f.write(json.dumps(mod))

    return




sotu = load_clean_sotu(file_name = './sotu_parsed.json')
# for i in range(4, 6):
#     lda_pipeline(sotu, i)


print len(sotu)
# res = lda_pipeline(sotu, 4)
res = cluster_pipeline(sotu, 5)
print type(res)
print res[0]
