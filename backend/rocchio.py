
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def rocchio(query, relevant, irrelevant, input_doc_matrix, coffee_name_to_index, a=.3, b=.3, c=.8, clip = True):
 
  mov_idx = coffee_name_to_index[query]
  q = input_doc_matrix[mov_idx]

  pt1 = a * q 

  rel_vec = np.zeros(len(q))
  irrel_vec = np.zeros(len(q))

  for rel_mov in set(relevant):
    rel_vec  += input_doc_matrix[ coffee_name_to_index[rel_mov]]
  
  if len(relevant)==0:
    pt2 = 0
  else:
    pt2 = b *  rel_vec * (1/len(relevant)) 

  for irrel_mov in set(irrelevant):
    irrel_vec += input_doc_matrix[coffee_name_to_index[irrel_mov]]

  if len(irrelevant)==0:
    pt3 = 0
  else:
    pt3 =c *  irrel_vec * (1/len(irrelevant))
  
  q_updated = pt1 + pt2 - pt3

  if clip:
    return np.clip(q_updated, 0, None)
  else:
    return q_updated


