from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('microchip_tests.txt',
                   header=None, names = ('test1','test2','released'))
X = data.values[:,:2]
y = data.values[:,2]

plt.scatter(X[y == 1, 0], X[y == 1, 1], c='green', label='Выпущен')
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', label='Бракован')

plt.title('2 теста микрочипов')
plt.legend()

def plot_boundary(clf, X, y, grid_step = .01, poly_featurizer = None):
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))
    Z = clf.predict(poly_featurizer.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)

poly = PolynomialFeatures(degree=7)
X_poly = poly.fit_transform(X)

C = 10000
logit = LogisticRegression(C=C, n_jobs=-1, random_state=17)
logit.fit(X_poly, y)

plot_boundary(logit,X,y,grid_step= .01, poly_featurizer=poly)

# plt.scatter(X[y == 1, 0], X[y == 1, 1], c='green', label='Выпущен')
# plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', label='Бракован')
#
# plt.title('2 теста микрочипов. Логит с C=0.01')
# plt.legend()

print("Доля правильных ответов классификатора на обучающей выборке:",
      round(logit.score(X_poly, y), 3))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
c_values = np.logspace(-2,3,500)

logit_seacher = LogisticRegressionCV(Cs = c_values, cv = skf, verbose= 1, n_jobs= -1)
logit_seacher.fit(X_poly,y)


# plt.plot(c_values, np.mean(logit_seacher.scores_[1], axis=0))
plt.show()