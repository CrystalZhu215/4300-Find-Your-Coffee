from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
from rocchio import rocchio

def perform_SVD(documents, query, relevant, irrelevant, coffee_name_to_index):

    vectorizer = TfidfVectorizer(stop_words = 'english', min_df = 70, max_df = 0.7)
    td_matrix = vectorizer.fit_transform([x[-2] for x in documents]) # combined description is last column in df
    print("20th entry", documents[0][-4])
    print("link", documents[0][-1])
    print("td matrix", td_matrix.shape)
    print("relevant", len(relevant))
    
    docs_compressed, s, words_compressed = svds(td_matrix, k=20)
    words_compressed = words_compressed.transpose()
    docs_compressed_normed = normalize(docs_compressed)

    query_tfidf = vectorizer.transform([query]).toarray()
    query_tfidf_new = rocchio(query_tfidf.flatten(), relevant, irrelevant, td_matrix.toarray(), coffee_name_to_index)
    query_vec = normalize(np.dot(query_tfidf_new.reshape(1, -1), words_compressed)).squeeze().T
    print("FINAL query vector", query_vec.shape)

    def closest_docs_to_query(query_vec_in):
        sims = docs_compressed_normed.dot(query_vec_in)
        asort = np.argsort(-sims)
        return [(i, documents[i][4], documents[i][3], documents[i][-4], documents[i][-1], sims[i]) for i in asort[1:]]

    return closest_docs_to_query(query_vec)