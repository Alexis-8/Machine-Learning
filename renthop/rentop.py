import json
import pandas as pd
import numpy as np
import warnings
from functools import reduce
from scipy.spatial.distance import euclidean
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")

with open("train.json", "r") as row_data:
    data = json.load(row_data)
    df = pd.DataFrame(data)

# Bag of words:

texts = [['i', 'have', 'a', 'cat'],
        ['he', 'have', 'a', 'dog'],
        ['he', 'and', 'i', 'have', 'a', 'cat', 'and', 'a', 'dog']]

dictionary = list(enumerate(set(list(reduce(lambda x, y: x + y, texts)))))

# Sparse matrix:

def vectorize(text):
    vector = np.zeros(len(dictionary))
    for i, word in dictionary:
        num = 0
        for w in text:
            if w == word:
                num += 1
        if num:
            vector[i] = num
    return vector

for t in texts:
    print(vectorize(t))

# Count Vectorizer:

vect1 = CountVectorizer(ngram_range=(1,1))
v1 = vect.fit_transform(['no i have cows', 'i have no cows']).toarray() #[[1 1 1]
                                                                            # [1 1 1]]
vect1.vocabulary_ # {'no': 2, 'have': 1, 'cows': 0}

vect2 = CountVectorizer(ngram_range=(1,2))
v2 = vect2.fit_transform(['no i have cows', 'i have no cows']).toarray() # [[1 1 1 0 1 0 1]
                                                                                # [1 1 0 1 1 1 0]]

vect3 = CountVectorizer(ngram_range=(3,3), analyzer="char_wb")

n1, n2, n3, n4 = vect3.fit_transform(['andersen', 'petersen', 'petrov', 'smith']).toarray()

