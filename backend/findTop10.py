import csv
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import helpers as analysis
from numpy import linalg as LA
import numpy as np

df = pd.read_csv("backend/data/coffee_fix.csv")


def findTopTen(user_query):
    """
    Takes a user query and returns an array of (dictionary, integer) pairs
    Dictionary is a dictionary of 'name', and 'description' of a coffee, and the integer is the cosine similarity

    Result is returned in order of cosine similarity highest to lowest
    User query must be input as a string
    """

    combined_descriptions = df[["desc_1"]].apply(lambda x: " ".join(x.dropna()), axis=1)
    combined_names = df[["name"]].apply(lambda x: " ".join(x.dropna()), axis=1)
    combined_locs = df[["origin"]].apply(lambda x: " ".join(x.dropna()), axis=1)

    combined_descriptions = [x for x in combined_descriptions]
    combined_names = [
        x + " from " + combined_locs[i] for i, x in enumerate(combined_names)
    ]

    vectorizer = TfidfVectorizer()

    # Replace query with the user query here
    query = [user_query]

    doc_vectors = vectorizer.fit_transform(query + combined_descriptions).toarray()
    # don't think we need these, but in the event that we have a very slow query, we can use this
    # index_to_vocab = {i: v for i, v in enumerate(vectorizer.get_feature_names_out())}
    # doc_to_index = {v: i for i, v in enumerate(combined_names)}
    index_to_doc_descriptions = {
        i: {"name": v, "description": combined_descriptions[i]}
        for i, v in enumerate(combined_names)
    }

    cosineSims = np.dot(doc_vectors[0], np.transpose(doc_vectors[1:])) / (
        LA.norm(doc_vectors[0]) * LA.norm(doc_vectors[1:])
    )

    cosineSims = [(x, i) for i, x in enumerate(cosineSims)]

    cosineSims = sorted(cosineSims, key=lambda x: x[0], reverse=True)

    topTen = cosineSims[:10]
    answer = []
    for sim, index in topTen:
        answer.append((index_to_doc_descriptions[index], sim))

    return answer
