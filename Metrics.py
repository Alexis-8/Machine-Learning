from sklearn import model_selection, linear_model, metrics, datasets
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

reg_data, reg_target = datasets.make_regression(n_features=2, n_informative=1, n_targets=1,
                                                noise=5, random_state=7)
clf_data, clf_target = datasets.make_classification(n_features=2, n_informative=2, n_classes=2,
                                                    n_redundant=0, n_clusters_per_class=1, random_state=7)

colors = ListedColormap(['red', 'blue'])
plt.subplot(211)
plt.scatter(list(map(lambda x: x[0], clf_data)), list(map(lambda x: x[1], clf_data)), c = clf_target, cmap=colors)

# plt.scatter(list(map(lambda x: x[0], reg_data)),reg_target, color = 'r')
# plt.scatter(list(map(lambda x: x[1], reg_data)), reg_target, color = 'b')

clf_train_data, clf_test_data, clf_train_labels, clf_test_lables = model_selection.train_test_split(clf_data,
                                                                                                    clf_target,
                                                                                                    test_size=0.3, random_state=1)
reg_train_data, reg_test_data, reg_train_labels, reg_test_lables = model_selection.train_test_split(reg_data,
                                                                                                    reg_target,
                                                                                                    test_size=0.3,
                                                                                                    random_state=1)
# classification:

classifier = linear_model.SGDClassifier(loss='log', random_state=1)
classifier.fit(clf_train_data, clf_train_labels)

predictions = classifier.predict(clf_test_data)
pre_probability = classifier.predict_proba(clf_test_data)

# realization of accuracy:

acc = sum([1. if pair[0] == pair[1] else 0. for pair in zip(clf_test_lables, predictions)])/len(clf_test_lables) # 0.93
acc_met = metrics.accuracy_score(clf_test_lables, predictions) # 0.93

conf_matr = metrics.confusion_matrix(clf_test_lables, predictions)

count_of_all_true_obj = sum([1 if pair[0] == pair[1] else 0 for pair in zip(clf_test_lables, predictions)]) # 28
matrix_count = conf_matr.diagonal().sum() # 28

# precision:

precis_0_class = metrics.precision_score(clf_test_lables, predictions, pos_label=0) # 0.94
precis_1_class = metrics.precision_score(clf_test_lables, predictions) # 0.917

# recall:

recall_0_class = metrics.recall_score(clf_test_lables, predictions, pos_label=0) # 0.94
recall_1_class = metrics.recall_score(clf_test_lables, predictions) # 0.91

clf_report = metrics.classification_report(clf_test_lables,predictions)

# ROC Curve:

fpr, tpr, thresholds = metrics.roc_curve(clf_test_lables, pre_probability[:,1])
plt.subplot(212)
plt.plot(fpr, tpr, label = 'linear_model')
plt.plot([0,1], "--", color = "grey", label = "random")
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("False Positiv Rate")
plt.ylabel("True Positiv Rate")
plt.title("ROC curve")
plt.legend(loc = "lower right")

AUC1 = metrics.roc_auc_score(clf_test_lables, predictions) # 0.93
AUC2 = metrics.roc_auc_score(clf_test_lables, pre_probability[:,1]) # 0.99

AvPS = metrics.average_precision_score(clf_test_lables, predictions) # 0.8736

log_loss = metrics.log_loss(clf_test_lables, pre_probability[:,1]) # 0.2177

# regression:

regressor = linear_model.SGDRegressor(random_state=1, n_iter_no_change=20)
regressor.fit(reg_train_data, reg_train_labels)
predictor = regressor.predict(reg_test_data)

# estimations:

MAE = metrics.mean_absolute_error(reg_test_lables, predictor) # 3.7267
MSE = metrics.mean_squared_error(reg_test_lables,predictor) # 23.526

R2 = metrics.r2_score(reg_test_lables, predictor) # 0.989578

plt.show()
