from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

df = pd.read_csv("data/coffee_fix.csv")
df['desc_all'] = df['desc_1'] + '\n' + df['desc_2'] + '\n' + df['desc_3']
df['desc_all'] = df['desc_all'].astype(str)

documents = df.values.tolist()

print(df.info())

vectorizer = TfidfVectorizer(stop_words = 'english', min_df = 70, max_df = 0.7)
td_matrix = vectorizer.fit_transform([x[-1] for x in documents]) # combined description is last column in df

docs_compressed, s, words_compressed = svds(td_matrix, k=20)
words_compressed = words_compressed.transpose()

word_to_index = vectorizer.vocabulary_
index_to_word = {i:t for t,i in word_to_index.items()}
words_compressed_normed = normalize(words_compressed, axis = 1)

# cosine similarity
def closest_words(word_in, words_representation_in, k = 10):
    if word_in not in word_to_index: return "Not in vocab."
    sims = words_representation_in.dot(words_representation_in[word_to_index[word_in],:])
    asort = np.argsort(-sims)[:k+1]
    return [(index_to_word[i],sims[i]) for i in asort[1:]]

td_matrix_np = td_matrix.transpose().toarray()
td_matrix_np = normalize(td_matrix_np)

# For words

word = 'citrus'
print("Query:", word)
print()
try:
    for w, sim in closest_words(word, words_compressed_normed):
        print("{}, {:.3f}".format(w, sim))
except:
    print("word not found")
print()

# For docs

docs_compressed_normed = normalize(docs_compressed)

def closest_docs(doc_index_in, doc_repr_in, k = 5):
    sims = doc_repr_in.dot(doc_repr_in[doc_index_in,:])
    asort = np.argsort(-sims)[:k+1]
    # Index 4 is name
    return [(documents[i][4], documents[i][-1], sims[i]) for i in asort[1:]]

for i in range(5):
    print("INPUT NAME: "+documents[i][4])
    print("INPUT DESCRIPTION: "+documents[i][-1])
    print()
    print("CLOSEST DESCRIPTIONS:")
    print("Using SVD:")
    for name, desc, score in closest_docs(i, docs_compressed_normed):
        print("{}: {:.3f} \n {}".format(name, score, desc))
        print()
    print("--------------------------------------------------------\n")

# Word to doc

def closest_docs_to_word(word_in, k = 5):
    if word_in not in word_to_index: return "Not in vocab."
    sims = docs_compressed_normed.dot(words_compressed_normed[word_to_index[word_in],:])
    asort = np.argsort(-sims)[:k+1]
    return [(i, documents[i][4], documents[i][-1], sims[i]) for i in asort[1:]]

for i, name, desc, sim in closest_docs_to_word("fruity"):
    print("{}\n {}\n {}\n {:.4f}".format(i, name, desc, sim))

print ("-------------------")

# Query to doc

query = "Chocolaty and nutty with floral notes, creamy mouthfeel and a hint of tartness"
query_tfidf = vectorizer.transform([query]).toarray()
query_vec = normalize(np.dot(query_tfidf, words_compressed)).squeeze()

def closest_docs_to_query(query_vec_in, k = 5):
    sims = docs_compressed_normed.dot(query_vec_in)
    asort = np.argsort(-sims)[:k+1]
    return [(i, documents[i][4], documents[i][-1],sims[i]) for i in asort[1:]]

for i, name, desc, sim in closest_docs_to_query(query_vec)[:10]:
    print("({}\n {}\n {}\n {:.4f}".format(i, name, desc, sim))


def top_10_from_query(query):
    query_tfidf = vectorizer.transform([query]).toarray()
    query_vec = normalize(np.dot(query_tfidf, words_compressed)).squeeze()

    return closest_docs_to_query(query_vec)