
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def rocchio(query_vec, relevant, irrelevant, input_doc_matrix, coffee_name_to_index, a=.3, b=.3, c=.8, clip = True):

  print("query vector", query_vec.shape)
  print("input doc matrix", type(input_doc_matrix))

  pt1 = a * query_vec

  rel_vec = np.zeros(len(query_vec))
  irrel_vec = np.zeros(len(query_vec))

  for rel_coffee in set(relevant):
    rel_vec += input_doc_matrix[coffee_name_to_index[rel_coffee]]

  print("rel vec", rel_vec.shape)
  
  if len(relevant)==0:
    pt2 = 0
  else:
    pt2 = b *  rel_vec * (1/len(relevant)) 

  for irrel_coffee in set(irrelevant):
    irrel_vec += input_doc_matrix[coffee_name_to_index[irrel_coffee]]


  print("irrel vec", irrel_vec.shape)

  if len(irrelevant)==0:
    pt3 = 0
  else:
    pt3 =c *  irrel_vec * (1/len(irrelevant))
  
  print("parts", (pt1 + pt2).shape)
  q_updated = pt1 + pt2 - pt3

  print("q updated", q_updated.shape)

  if clip:
    return np.clip(q_updated, 0, None)
  else:
    return q_updated


