from sklearn.datasets import load_files
from sklearn.feature_extraction import text
from sklearn import linear_model, metrics
import pandas as pd
import numpy as np
from scipy import sparse


pd_data = pd.read_csv("IMDB Dataset.csv")

data = pd_data.drop(['sentiment'], axis=1)
labels = pd_data['sentiment'].map({"negative": 0, "positive": 1})

train_part_size = int(.5*data.shape[0])
train_data, train_labels = data[:train_part_size], labels[:train_part_size]
test_data, test_labels =  data[train_part_size:],  labels[train_part_size:]

# BOW - Bag of Words:

a = np.zeros([5,5])
a[0,3] = 1
a[1,4] = 6
a[2,3] = 3
a[4,0] = 2
a[4,1] = 6

a_pd = pd.DataFrame(a, columns = ["apple", "girl","wax","luck", "sadness"])
b = sparse.csr_matrix(a_pd)

cv = text.CountVectorizer()
x_train_sparse = cv.fit_transform(train_data["review"])

cv.vocabulary_ # 'napoleon': 45847, 'holloway': 31977, 'winchell': индексы слов

x_test_sparse = cv.transform(test_data["review"])

# accuracy - адекватная метрика? как распределены плохие и хорошие отзывы?

np.bincount(train_labels), np.bincount(test_labels) # [12526 12474] [12474 12526]

logit = linear_model.LogisticRegression(random_state=3, n_jobs=-1)
sgd_classifier = linear_model.SGDClassifier(max_iter=40, random_state=3, n_jobs=-1) # 40 = 1000000/len(x_train_sparse)

logit.fit(x_train_sparse, train_labels)
sgd_classifier.fit(x_train_sparse,train_labels)

logit_acc = metrics.accuracy_score(test_labels, logit.predict(x_test_sparse)) # 0.88568
sgd_acc = metrics.accuracy_score(test_labels, sgd_classifier.predict(x_test_sparse)) # 0.8714

