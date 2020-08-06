from matplotlib.colors import ListedColormap
from sklearn import model_selection, datasets, metrics, linear_model
import matplotlib.pyplot as plt

import numpy as np

data, target, coef = datasets.make_regression(n_features=3, n_informative=1, n_targets=1,
                                              noise=5., coef=True, random_state=2)
plt.scatter(list(map(lambda x: x[0], data)), target, color = 'red')
plt.scatter(list(map(lambda x: x[1], data)), target, color = 'blue')
plt.scatter(list(map(lambda x: x[2], data)), target, color = 'g')

train_data, test_data, train_labels, test_labels = model_selection.train_test_split(data, target,
                                                                                    test_size=0.3)

# linear_regression:

linear_regresor = linear_model.LinearRegression()
linear_regresor.fit(train_data, train_labels)
predictions = linear_regresor.predict(test_data)

mean_absolute_error = metrics.mean_absolute_error(test_labels,predictions) # error = 4.88219
linear_scoring = model_selection.cross_val_score(linear_regresor, data, target, scoring="neg_mean_absolute_error", cv = 10)
print("mean: {}, std: {}".format(linear_scoring.mean(), linear_scoring.std()))

scorer = metrics.make_scorer(metrics.mean_absolute_error, greater_is_better=True)
linear_scoring2 = model_selection.cross_val_score(linear_regresor, data, target, scoring=scorer, cv = 10)
print("mean: {}, std: {}".format(linear_scoring2.mean(), linear_scoring2.std()))

print(coef)
print(linear_regresor.coef_)


# another Lasso-regression:

lasso_regressor = linear_model.Lasso(random_state=3)
lasso_regressor.fit(train_data,train_labels)
lasso_predictor = lasso_regressor.predict(test_data)

lasso_scoring = model_selection.cross_val_score(lasso_regressor, data, target, scoring=scorer)
print("mean: {}, std: {}".format(lasso_scoring.mean(), lasso_scoring.std()))

plt.show()
