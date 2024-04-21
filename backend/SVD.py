from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds

def perform_SVD(documents, query):

    vectorizer = TfidfVectorizer(stop_words = 'english', min_df = 70, max_df = 0.7)
    td_matrix = vectorizer.fit_transform([x[-1] for x in documents]) # combined description is last column in df

    docs_compressed, s, words_compressed = svds(td_matrix, k=20)
    words_compressed = words_compressed.transpose()
    docs_compressed_normed = normalize(docs_compressed)

    def closest_docs_to_query(query_vec_in):
        sims = docs_compressed_normed.dot(query_vec_in)
        asort = np.argsort(-sims)
        return [(i, documents[i][4], documents[i][3], documents[i][-1], sims[i]) for i in asort[1:]]

    query_tfidf = vectorizer.transform([query]).toarray()
    query_vec = normalize(np.dot(query_tfidf, words_compressed)).squeeze()

    return closest_docs_to_query(query_vec)