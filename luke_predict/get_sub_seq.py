import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_tfidf(subwords, top_subword=None, tfidf=False):
    if top_subword is not None:
        subword = [s["input_ids"] for s in subwords] + [top_subword]
    else:
        subword = [s["input_ids"] for s in subwords]
    if tfidf:
        vectorizer = TfidfVectorizer(analyzer=lambda x: x)
    else:
        vectorizer = CountVectorizer(analyzer=lambda x: x)
    bow = vectorizer.fit_transform(subword).toarray()
    return bow


def cos_sim_choice(subwords, top_subword, out, tfidf=False):
    bow = get_tfidf(subwords, top_subword, tfidf=tfidf)
    selected_subwords_id = [-1]
    cos_sims = np.zeros(bow.shape[0] - 1)
    cos = cosine_similarity(bow, bow[:-1])
    for i in range(out):
        cos_sims = cos_sims + cos[selected_subwords_id[-1]]
        min_sims = np.argmin(cos_sims)

        selected_subwords_id.append(min_sims)
    selected_subwords = [subwords[id] for id in selected_subwords_id[1:]]
    return selected_subwords


def kmeans_choice(subwords, out, tfidf=False):
    bow = get_tfidf(subwords, tfidf=tfidf)
    bow = np.unique(bow, axis=0)
    ids = np.arange(bow.shape[0])
    if bow.shape[0] == 1:
        return subwords[:out]
    kmeans = KMeans(
        n_clusters=min(out, bow.shape[0]), init="k-means++", random_state=0, n_init=min(out, bow.shape[0]), max_iter=300
    )
    clusters = kmeans.fit(bow).labels_

    selected_subwords = []
    for o in range(out):
        cluster = bow[clusters == o]
        cluster_ids = ids[clusters == o]
        if cluster.shape[0] == 0:
            selected_subwords.append(subwords[0])
        else:
            ave = cluster.mean()
            dis = [np.linalg.norm(c - ave) for c in cluster]
            min_id = cluster_ids[np.argmin(dis)]
            selected_subwords.append(subwords[min_id])

    return selected_subwords


def get_subword_sequences(tokenizer, texts, entity_spans, n=500, out=10, method="random", tfidf=True):

    if method == "random":
        selected_subwords = [tokenizer(texts, entity_spans=entity_spans) for _ in range(out)]
    else:
        subwords = [tokenizer(texts, entity_spans=entity_spans) for _ in range(n)]
        tokenizer.const_tokenize()
        top_subword = tokenizer(texts, entity_spans=entity_spans)["input_ids"]
        tokenizer.random_tokenize()
        if method == "cos-sim":
            selected_subwords = cos_sim_choice(subwords, top_subword, out, tfidf=tfidf)
        elif method == "k-means":
            selected_subwords = kmeans_choice(subwords, out, tfidf=tfidf)
        else:
            raise ValueError(f"Invalid value\nmethod have to be random|cos-sim|k-means. You set {method}")

    return selected_subwords
