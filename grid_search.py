from sklearn import linear_model, model_selection, datasets, svm

import pandas as pd
import numpy as np

iris = datasets.load_iris()

train_data, test_data, train_labels, test_labels = model_selection.train_test_split(iris.data, iris.target, test_size=0.3,
                                                                                    random_state=0)
classifire = linear_model.SGDClassifier(random_state=0)
classifire.get_params().keys() # dict_keys(['alpha', 'average', 'class_weight', 'early_stopping', 'epsilon', 'eta0', 'fit_intercept', 'l1_ratio', 'learning_rate', 'loss', 'max_iter', 'n_iter_no_change', 'n_jobs', 'penalty', 'power_t', 'random_state', 'shuffle', 'tol', 'validation_fraction', 'verbose', 'warm_start'])
parameters_grid = {'loss': ["hinge", "log", "squared_hinge", "squared_loss"],
                   'penalty': ["l1","l2"],
                  'n_iter_no_change' : range(5,10),
                   'alpha': np.linspace(0.0001,0.001,num=5)}
cv = model_selection.StratifiedShuffleSplit(n_splits=10,  test_size=0.2, random_state=0)
grid_cv = model_selection.GridSearchCV(classifire,parameters_grid, scoring="accuracy", cv = cv)

grid_cv.fit(train_data, train_labels)
print(grid_cv.best_estimator_)
grid_cv.best_score_ # 0.9857142857142858
grid_cv.best_params_ #  {'alpha': 0.00055, 'loss': 'hinge', 'n_iter_no_change': 6, 'penalty': 'l1'}

randomized_search_cv = model_selection.RandomizedSearchCV(classifire, parameters_grid,scoring="accuracy",n_iter=20, cv = cv)
randomized_search_cv.fit(train_data, train_labels)
print(randomized_search_cv.best_score_) # 0.9666666666666666
print(randomized_search_cv.best_params_) # {'penalty': 'l1', 'n_iter_no_change': 9, 'loss': 'hinge', 'alpha': 0.001}
