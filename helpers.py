from typing import List, Tuple, Dict
from collections.abc import Callable
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from numpy import linalg as LA
import json
import math


def average_precision(ranking_in, relevant):
    rel_rank = sorted([ranking_in.index(r) + 1 for r in relevant])
    return np.mean([(i + 1) * 1.0 / (r) for i, r in enumerate(rel_rank)])


def build_vectorizer(max_features, stop_words, max_df=0.8, min_df=10, norm="l2"):
    """Returns a TfidfVectorizer object with the above preprocessing properties.

    Note: This function may log a deprecation warning. This is normal, and you
    can simply ignore it.

    Parameters
    ----------
    max_features : int
        Corresponds to 'max_features' parameter of the sklearn TfidfVectorizer
        constructer.
    stop_words : str
        Corresponds to 'stop_words' parameter of the sklearn TfidfVectorizer constructer.
    max_df : float
        Corresponds to 'max_df' parameter of the sklearn TfidfVectorizer constructer.
    min_df : float
        Corresponds to 'min_df' parameter of the sklearn TfidfVectorizer constructer.
    norm : str
        Corresponds to 'norm' parameter of the sklearn TfidfVectorizer constructer.

    Returns
    -------
    TfidfVectorizer
        A TfidfVectorizer object with the given parameters as its preprocessing properties.
    """
    return TfidfVectorizer(
        max_features=max_features,
        stop_words=stop_words,
        max_df=max_df,
        min_df=min_df,
        norm=norm,
    )


def get_sim(mov1, mov2, input_doc_mat, input_movie_name_to_index):
    """Returns a float giving the cosine similarity of
       the two movie transcripts.

    Params: {mov1 (str): Name of the first movie.
             mov2 (str): Name of the second movie.
             input_doc_mat (numpy.ndarray): Term-document matrix of movie transcripts, where
                    each row represents a document (movie transcript) and each column represents a term.
             movie_name_to_index (dict): Dictionary that maps movie names to the corresponding row index
                    in the term-document matrix.}
    Returns: Float (Cosine similarity of the two movie transcripts.)
    """
    # TODO-5.2
    first, second = (
        input_doc_mat[input_movie_name_to_index[mov1]],
        input_doc_mat[input_movie_name_to_index[mov2]],
    )

    return np.dot(first, second) / (LA.norm(first) * LA.norm(second))


def top_terms(movs, input_doc_mat, index_to_vocab, movie_name_to_index, top_k=10):
    """Returns a list of the top k similar terms (in order) between the
        inputted movie transcripts.

    Parameters
    ----------
    movs : str list (Length >= 2)
        List of movie names
    input_doc_mat : np.ndarray
        The term document matrix of the movie transcripts. input_doc_mat[i][j] is the tfidf
        of the movie i for the word j.
    index_to_vocab : dict
         A dictionary linking the index of a word (Key: int) to the actual word (Value: str).
         Ex: {0: 'word_0', 1: 'word_1', .......}
    movie_name_to_index : dict
         A dictionary linking the movie name (Key: str) to the movie index (Value: int).
         Ex: {'movie_0': 0, 'movie_1': 1, .......}
    top_k : int
        The k in the top k similar words to be returned. Ex: If top_k = 8, return top 8 similar words

    Returns
    -------
    list
        A list of the top k similar terms (in order) between the inputted movie transcripts
    """
    # TODO-5.3
    result = np.ones(len(input_doc_mat[0]))

    for mov in movs:
        result = np.multiply(result, input_doc_mat[movie_name_to_index[mov]])
    indexes = np.argsort(result)
    revIndexes = indexes[::-1]

    top_k_indexes = revIndexes[:top_k]

    words = []
    for index in top_k_indexes:
        words.append(index_to_vocab[index])

    return words


def build_movie_sims_cos(
    n_mov, movie_index_to_name, input_doc_mat, movie_name_to_index, input_get_sim_method
):
    """Returns a movie_sims matrix of size (num_movies,num_movies) where for (i,j):
        [i,j] should be the cosine similarity between the movie with index i and the movie with index j

    Note: You should set values on the diagonal to 1
    to indicate that all movies are trivially perfectly similar to themselves.

    Params: {n_mov: Integer, the number of movies
             movie_index_to_name: Dictionary, a dictionary that maps movie index to name
             input_doc_mat: Numpy Array, a numpy array that represents the document-term matrix
             movie_name_to_index: Dictionary, a dictionary that maps movie names to index
             input_get_sim_method: Function, a function to compute cosine similarity}
    Returns: Numpy Array
    """
    # TODO-5.4
    result = np.ndarray(shape=(n_mov, n_mov))

    for i in range(n_mov):
        result[i][i] = 1

    for i in range(n_mov):
        for j in range(i + 1, n_mov):
            cosSim = input_get_sim_method(
                movie_index_to_name[i],
                movie_index_to_name[j],
                input_doc_mat,
                movie_name_to_index,
            )
            result[i][j], result[j][i] = cosSim, cosSim
    return result


def build_movie_sims_jac(n_mov, input_data):
    """Returns a movie_sims_jac matrix of size (num_movies,num_movies) where for (i,j) :
        [i,j] should be the jaccard similarity between the category sets for movies i and j
        such that movie_sims_jac[i,j] = movie_sims_jac[j,i].

    Note:
        Movies sometimes contain *duplicate* categories! You should only count a category once

        A movie should have a jaccard similarity of 1.0 with itself.

    Params: {n_mov: Integer, the number of movies,
            input_data: List<Dictionary>, a list of dictionaries where each dictionary
                     represents the movie_script_data including the script and the metadata of each movie script}
    Returns: Numpy Array
    """
    # TODO-5.5
    result = np.ndarray(shape=(n_mov, n_mov))

    for i in range(n_mov):
        result[i][i] = 1.0
    for i in range(n_mov):
        for j in range(i + 1, n_mov):
            first, second = set(input_data[i]["categories"]), set(
                input_data[j]["categories"]
            )
            jacSim = len(first.intersection(second)) / len(first.union(second))
            result[i][j], result[j][i] = jacSim, jacSim
    return result


def precision_recall(ranking_in, relevant):
    """
    Returns lists of precision and recall at different k values

    Parameters
    ----------
    ranking_in : str list
        List with sorted ranking of movies (movie names), starting with the most similar, and ending
        with the least similar.
    relevant : str list
        List of movies (movie names) relevant to the original query

    Returns
    -------
    tuple: (np.ndarray, np.ndarray)
        Returns tuple such that tuple[0] is numpy array of precision at different k values and
        tuple[1] is numpy array of recall at different k values.

        tuple[0] -> precision: numpy array of length equal to the length+1 of ranking_in, where
        precision[k] = the precision@k. Leave precision[0] to be 0.

        tuple[1] -> recall: numpy array of length equal to the length+1 of ranking_in, where
        recall[k] = the recall@k. Leave recall[0] to be 0.
    """
    # TODO-5.6

    def isRelevant(item):
        if item in relevant:
            return True

    def precision(Sys):
        return len([item for item in Sys if isRelevant(item)]) / len(Sys)

    def recall(Sys):
        return len([item for item in Sys if isRelevant(item)]) / len(relevant)

    def precision_at_k(Sys, k):
        return precision(Sys[:k])

    def recall_at_k(Sys, k):
        return recall(Sys[:k])

    precisions, recalls = np.zeros(shape=len(ranking_in) + 1), np.zeros(
        shape=len(ranking_in) + 1
    )
    precisions[0], recalls[0] = 0, 0

    for i in range(1, len(ranking_in) + 1):
        precisions[i], recalls[i] = precision_at_k(ranking_in, i), recall_at_k(
            ranking_in, i
        )

    return (precisions, recalls)


def compute_fscore(precision, recall):
    """
    Returns lists of f-score values at different k values, where fscore[k] = the fscore@k

    Parameters
    ----------
    precision : np.ndarray
        numpy array of precision values at different k values, where precision[k] = the
        precision@k.
    recall : np.ndarray
        numpy array of recall values at different k values, where recall[k] = the
        recall@k.

    Returns
    -------
    np.ndarray
        Returns a numpy array of length equal to the length of the precision (and recall) parameter
        array, where fscore[k] = the fscore@k.
    """
    # TODO-5.7
    result = np.zeros(shape=len(precision))
    for i in range(1, len(precision)):
        if precision[i] != 0 and recall[i] != 0:
            result[i] = 2 / (1 / precision[i]) + (1 / recall[i])
        else:
            result[i] = 0
    return result


def rocchio(
    query,
    relevant,
    irrelevant,
    input_doc_matrix,
    movie_name_to_index,
    a=0.3,
    b=0.3,
    c=0.8,
    clip=True,
):
    """Returns a vector representing the modified query vector.

    Note:
        If the `clip` parameter is set to True, the resulting vector should have
        no negatve weights in it!

        Also, be sure to handle the cases where relevant and irrelevant are empty lists.

    Params: {query: String (the name of the movie being queried for),
             relevant: List (the names of relevant movies for query),
             irrelevant: List (the names of irrelevant movies for query),
             input_doc_matrix: Numpy Array,
             movie_name_to_index: Dict,
             a,b,c: floats (weighting of the original query, relevant queries,
                             and irrelevant queries, respectively),
             clip: Boolean (whether or not to clip all returned negative values to 0)}
    Returns: Numpy Array
    """
    # TODO-5.8
    queryVector = input_doc_matrix[movie_name_to_index[query]]
    queryVector = queryVector * a
    relevantCentroid, irrelevantCentroid = 0, 0
    if relevant:
        arrayRelevent = []
        for i, movieTitle in enumerate(relevant):
            arrayRelevent.append(input_doc_matrix[movie_name_to_index[movieTitle]])
            relevantCentroid = np.sum(arrayRelevent, axis=0) / len(arrayRelevent)
        queryVector = queryVector + relevantCentroid * b
    if irrelevant:
        arrayIrrelevant = []
        for movieTitle in irrelevant:
            arrayIrrelevant.append(input_doc_matrix[movie_name_to_index[movieTitle]])
            irrelevantCentroid = np.sum(arrayIrrelevant, axis=0) / len(arrayIrrelevant)
        queryVector = queryVector - irrelevantCentroid * c

    if clip:
        result = np.where(queryVector < 0, 0, queryVector)

    return result


def top_10_with_rocchio(
    relevant_in,
    irrelevant_in,
    input_doc_matrix,
    movie_name_to_index,
    movie_index_to_name,
    input_rocchio,
):
    """Returns a dictionary in the following format:
    {
        'the matrix': [movie1,movie2,...,movie10],
        'star wars': [movie1,movie2,...,movie10],
        'a nightmare on elm street': [movie1,movie2,...,movie10]
    }

    Note:
        You can assume that relevant_in[i][0] = irrelevant_in[i][0]
        (i.e. the queries are in the same order).

        You should use the default rocchio parameters.

        You should NOT return the query itself in the list of most common
        movies.

    Parameters
    ----------
    relevant_in : (query: str, [relevant documents]: str list) list
        List of tuples of the form:
        tuple[0] = name of movie being queried (str),
        tuple[1] = list of names of the relevant movies to the movie being queried (str list).
    irrelevant_in : (query: str, [irrelevant documents]: str list) list
        The same format as relevant_in except tuple[1] contains list of irrelevant movies instead.
    input_doc_matrix : np.ndarray
        The term document matrix of the movie transcripts. input_doc_mat[i][j] is the tfidf
        of the movie i for the word j.
    movie_name_to_index : dict
         A dictionary linking the movie name (Key: str) to the movie index (Value: int).
         Ex: {'movie_0': 0, 'movie_1': 1, .......}
    movie_index_to_name : dict
         A dictionary linking the movie index (Key: int) to the movie name (Value: str).
         Ex: {0:'movie_0', 1:'movie_1', .......}
    input_rocchio: function
        A function implementing the rocchio algorithm. Refer to Q6 for the function
        input parameters. Make sure you use the function's default parameters as
        much as possible.

    Returns
    -------
    dict
        Returns the top ten highest ranked movies for each query in the format described above.
    """
    # TODO-5.9
    result = {}
    for i, x in enumerate(relevant_in):
        curCalculations = {}
        query, relevant = x[0], x[1]
        irrelevant = irrelevant_in[i][1]
        newRocchioVector = input_rocchio(
            query, relevant, irrelevant, input_doc_matrix, movie_name_to_index
        )

        for i in range(len(input_doc_matrix)):
            # skip if you are looking at the same movie as the query
            if i == movie_name_to_index[query]:
                continue
            cosSim = np.dot(newRocchioVector, input_doc_matrix[i]) / (
                LA.norm(newRocchioVector) * LA.norm(input_doc_matrix[i])
            )
            curCalculations[movie_index_to_name[i]] = cosSim

        res = list(
            sorted(curCalculations.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        res = [i[0] for i in res]
        result[query] = res
    return result


def mean_average_precision_rocchio(
    relevant_in,
    irrelevant_in,
    input_doc_matrix,
    movie_name_to_index,
    movie_index_to_name,
    input_rocchio,
):
    """Returns a float corresponding to the mean AP statistic for the Rocchio-updated input queries
        and the similarity matrix
    Note:
        You can assume that relevant_in[i][0] = irrelevant_in[i][0]
        (i.e. the queries are in the same order).

        You should use the default rocchio parameters.

        You should NOT include the query itself in the list of most common
        movies.

    Parameters
    ----------
    relevant_in : (query: str, [relevant documents]: str list) list
        List of tuples of the form:
        tuple[0] = name of movie being queried (str),
        tuple[1] = list of names of the relevant movies to the movie being queried (str list).
    irrelevant_in : (query: str, [irrelevant documents]: str list) list
        The same format as relevant_in except tuple[1] contains list of irrelevant movies instead.
    input_doc_matrix : np.ndarray
        The term document matrix of the movie transcripts. input_doc_mat[i][j] is the tfidf
        of the movie i for the word j.
    movie_name_to_index : dict
         A dictionary linking the movie name (Key: str) to the movie index (Value: int).
         Ex: {'movie_0': 0, 'movie_1': 1, .......}
    movie_index_to_name : dict
         A dictionary linking the movie index (Key: int) to the movie name (Value: str).
         Ex: {0:'movie_0', 1:'movie_1', .......}
    input_rocchio: function
        A function implementing the rocchio algorithm. Refer to Q6 for the function
        input parameters. Make sure you use the function's default parameters as
        much as possible.

    Returns
    -------
    float
        Returns a float corresponding to the mean AP statistic for the Rocchio-updated input queries
        and the similarity matrix
    """
    # TODO-5.10
    queries = []

    result = {}

    for query, relevant_docs in relevant_in:
        queries.append(query)
    ap = 0
    for i, x in enumerate(queries):
        curQuery = queries[i]
        rocchio = input_rocchio(
            curQuery,
            relevant_in[i][1],
            irrelevant_in[i][1],
            input_doc_matrix,
            movie_name_to_index,
        )
        reversedList = np.argsort(rocchio.dot(input_doc_matrix.T))[::-1]

        rankings = [r for r in reversedList if movie_name_to_index[curQuery] != r]

        inverse_rank = []

        for j, x in enumerate(rankings):
            movie = movie_index_to_name[rankings[j]]
            inverse_rank.append(movie)

        ap += average_precision(inverse_rank, relevant_in[i][1])
    ap_stat = ap / len(relevant_in)
    return ap_stat
