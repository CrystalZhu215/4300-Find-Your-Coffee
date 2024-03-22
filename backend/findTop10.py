def findTopTen(user_query):

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

    df = pd.read_csv("data/coffee_fix.csv")

    combined_descriptions = df[["desc_1"]].apply(lambda x: " ".join(x.dropna()), axis=1)
    combined_names = df[["name"]].apply(lambda x: " ".join(x.dropna()), axis=1)
    combined_locs = df[["origin"]].apply(lambda x: " ".join(x.dropna()), axis=1)

    combined_descriptions = [x for x in combined_descriptions]
    combined_names = [
        x + " from " + combined_locs[i] for i, x in enumerate(combined_names)
    ]

    vectorizer = TfidfVectorizer()
    doc_by_vocab = vectorizer.fit_transform(combined_descriptions).toarray()

    index_to_vocab = {i: v for i, v in enumerate(vectorizer.get_feature_names_out())}
    vocab_to_index = {index_to_vocab[i]: i for i in index_to_vocab}
    doc_to_index = {v: i for i, v in enumerate(combined_names)}

    # Replace query with the user query here

    query = user_query
    query = query.lower()
    query = query.split(" ")

    queryVector = [0] * doc_by_vocab.shape[1]

    for c in query:
        if c in vocab_to_index:
            queryVector[vocab_to_index[c]] += 1

    similarities = []
    for i in range(len(doc_by_vocab)):
        similarities.append(
            (
                np.dot(queryVector, doc_by_vocab[i])
                / (LA.norm(queryVector) * LA.norm(doc_by_vocab[i])),
                i,
            )
        )

    similarities.sort(reverse=True)
    topTen = similarities[:10]

    #Names = [(combined_names[i], combined_descriptions[i]) for x, i in topTen]
    return [{"coffee_name": combined_names[i], "description": combined_descriptions[i]} for _, i in topTen]