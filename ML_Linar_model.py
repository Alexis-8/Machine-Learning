from matplotlib.colors import ListedColormap
from sklearn import model_selection
from sklearn import datasets, metrics, linear_model
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt
import numpy as np

blobs = datasets.make_blobs(centers=2, cluster_std=5.5, random_state=1)
colors = ListedColormap(["red", "blue"])

plt.figure(figsize=(8,8))
plt.scatter(list(map(lambda x: x[0], blobs[0])), list(map(lambda x: x[1], blobs[0])), c=blobs[1], cmap=colors)

train_data, test_data, train_lables, test_labels = train_test_split(blobs[0], blobs[1], test_size=0.3,
                                                                    random_state=1)
# Create an object - classifier:

ridge_classifier = linear_model.RidgeClassifier(random_state=1)

# Training the classifier:

ridge_classifier.fit(train_data, train_lables)

# Use the classifier:

ridge_prediction = ridge_classifier.predict(test_data)

# Quality control:

metrics.accuracy_score(test_labels,ridge_prediction) # 0.8666666666666667  good!
print(ridge_classifier.coef_) # weights: [[-0.0854443  -0.07273219]]
print(ridge_classifier.intercept_) # coef before free member [-0.31250723]
plt.show()

# LogisticRegression:

log_regressor = linear_model.LogisticRegression(random_state=1)
log_regressor.fit(train_data,train_lables)

lr_predictions = log_regressor.predict(test_data)
lr_predictions_proba = log_regressor.predict_proba(test_data) # probability of prediction

# print(test_labels)
# print(lr_predictions)
# print(lr_predictions_proba)

accuracy_score = metrics.accuracy_score(test_labels,lr_predictions) # 0.8

# Quality control on cross-validation:

ridge_scoring1 = cross_val_score(ridge_classifier, blobs[0], blobs[1], scoring="accuracy", cv = 10)
lr_scoring = cross_val_score(log_regressor, blobs[0], blobs[1], scoring="accuracy", cv = 10)

print('Ridge mean: {}, ridge min = {}, ridge max = {}, ridge std: {}'.format(ridge_scoring1.mean(), ridge_scoring1.min(),
                                                              ridge_scoring1.max(), ridge_scoring1.std()))
print('Log mean: {}, ridge min = {}, ridge max = {}, log std: {}'.format(lr_scoring.mean(), lr_scoring.min(),
                                                              lr_scoring.max(), lr_scoring.std()))

# own accuracy_score:

scorer = metrics.make_scorer(metrics.accuracy_score)
cv_strategy = model_selection.StratifiedShuffleSplit(n_splits=20, test_size=0.3, random_state=2)

ridge_scoring = cross_val_score(ridge_classifier, blobs[0], blobs[1], scoring=scorer, cv = cv_strategy)
Log_scoring1 = cross_val_score(log_regressor, blobs[0], blobs[1], scoring=scorer, cv = cv_strategy)
print(ridge_scoring, Log_scoring1)