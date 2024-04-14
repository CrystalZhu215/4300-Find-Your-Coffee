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

vectorizer = TfidfVectorizer(stop_words = 'english', min_df = 70, max_df = 0.7)
td_matrix = vectorizer.fit_transform(df['desc_all'].to_list())

docs_compressed, s, words_compressed = svds(td_matrix, k=20)
words_compressed = words_compressed.transpose()

''' Show that most data lives in 20 dimensions
plt.plot(s[::-1])
plt.xlabel("Singular value number")
plt.ylabel("Singular value")
plt.show()
'''

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

documents = df['desc_all'].to_list()
def closest_docs(doc_index_in, doc_repr_in, k = 5):
    sims = doc_repr_in.dot(doc_repr_in[doc_index_in,:])
    asort = np.argsort(-sims)[:k+1]
    return [(documents[i],sims[i]) for i in asort[1:]]

for i in range(5):
    print("INPUT DESCRIPTION: "+documents[i])
    print()
    print("CLOSEST DESCRIPTIONS:")
    print("Using SVD:")
    for desc, score in closest_docs(i, docs_compressed_normed):
        print("{}:{:.3f}".format(desc, score))
        print()
    print("--------------------------------------------------------\n")

# Word to doc

def closest_docs_to_word(word_in, k = 5):
    if word_in not in word_to_index: return "Not in vocab."
    sims = docs_compressed_normed.dot(words_compressed_normed[word_to_index[word_in],:])
    asort = np.argsort(-sims)[:k+1]
    return [(i, documents[i],sims[i]) for i in asort[1:]]

for i, desc, sim in closest_docs_to_word("fruity"):
    print("{}\n {}\n {:.4f}".format(i, desc, sim))

print()
