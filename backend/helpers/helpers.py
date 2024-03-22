import os
from nltk.tokenize import TreebankWordTokenizer



def insertion_cost(message, j):
    return 1


def deletion_cost(query, i):
    return 1


def substitution_cost(query, message, i, j):
    if query[i - 1] == message[j - 1]:
        return 0
    else:
        return 1


def edit_matrix(query, message, ins_cost_func, del_cost_func, sub_cost_func):
    """Calculates the edit matrix

    Arguments
    =========

    query: query string,

    message: message string,

    ins_cost_func: function that returns the cost of inserting a letter,

    del_cost_func: function that returns the cost of deleting a letter,

    sub_cost_func: function that returns the cost of substituting a letter,

    Returns:
        edit matrix {(i,j): int}
    """

    m = len(query) + 1
    n = len(message) + 1

    chart = {(0, 0): 0}
    for i in range(1, m):
        chart[i, 0] = chart[i - 1, 0] + del_cost_func(query, i)
    for j in range(1, n):
        chart[0, j] = chart[0, j - 1] + ins_cost_func(message, j)
    for i in range(1, m):
        for j in range(1, n):
            chart[i, j] = min(
                chart[i - 1, j] + del_cost_func(query, i),
                chart[i, j - 1] + ins_cost_func(message, j),
                chart[i - 1, j - 1] + sub_cost_func(query, message, i, j),
            )
    return chart


adj_chars = [
    ("a", "q"),
    ("a", "s"),
    ("a", "z"),
    ("b", "g"),
    ("b", "m"),
    ("b", "n"),
    ("b", "v"),
    ("c", "d"),
    ("c", "v"),
    ("c", "x"),
    ("d", "c"),
    ("d", "e"),
    ("d", "f"),
    ("d", "s"),
    ("e", "d"),
    ("e", "r"),
    ("e", "w"),
    ("f", "d"),
    ("f", "g"),
    ("f", "r"),
    ("f", "v"),
    ("g", "b"),
    ("g", "f"),
    ("g", "h"),
    ("g", "t"),
    ("h", "g"),
    ("h", "j"),
    ("h", "m"),
    ("h", "n"),
    ("h", "y"),
    ("i", "k"),
    ("i", "o"),
    ("i", "u"),
    ("j", "h"),
    ("j", "k"),
    ("j", "u"),
    ("k", "i"),
    ("k", "j"),
    ("k", "l"),
    ("l", "k"),
    ("l", "o"),
    ("m", "b"),
    ("m", "h"),
    ("n", "b"),
    ("n", "h"),
    ("o", "i"),
    ("o", "l"),
    ("o", "p"),
    ("p", "o"),
    ("q", "a"),
    ("q", "w"),
    ("r", "e"),
    ("r", "f"),
    ("r", "t"),
    ("s", "a"),
    ("s", "d"),
    ("s", "w"),
    ("s", "x"),
    ("t", "g"),
    ("t", "r"),
    ("t", "y"),
    ("u", "i"),
    ("u", "j"),
    ("u", "y"),
    ("v", "b"),
    ("v", "c"),
    ("v", "f"),
    ("w", "e"),
    ("w", "q"),
    ("w", "s"),
    ("x", "c"),
    ("x", "s"),
    ("x", "z"),
    ("y", "h"),
    ("y", "t"),
    ("y", "u"),
    ("z", "a"),
    ("z", "x"),
]
