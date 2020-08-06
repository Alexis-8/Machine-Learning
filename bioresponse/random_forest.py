from sklearn import ensemble, linear_model, model_selection, metrics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

bioresponse = pd.read_csv("bioresponse.csv", header=0, sep=",")
bioresponse.shape # (3751, 1777)
bioresponse.columns # ['Activity', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9',
       #
       # 'D1767', 'D1768', 'D1769', 'D1770', 'D1771', 'D1772', 'D1773', 'D1774',
       # 'D1775', 'D1776']
bioresponse_target = bioresponse['Activity'].values
print("bioresponse = 1: {:.2f}\nbioresponse = 0: {:.2f}".format(sum(bioresponse_target)/float(len(bioresponse_target)),
                                                                1.0 - sum(bioresponse_target)/float(len(bioresponse_target))))

bioresponse_data = bioresponse.iloc[:,1:]

rf_classifuire_low_depth = ensemble.RandomForestClassifier(n_estimators=50, max_depth=2, random_state=3)
train_sizes, train_scores, test_scores = model_selection.learning_curve(rf_classifuire_low_depth, bioresponse_data, bioresponse_target,
                                                                        train_sizes=np.arange(0.1,1.,0.2),
                                                                        cv = 3, scoring="accuracy")
train_sizes # [ 250  750 1250 1750 2250]
train_scores_mean  = train_scores.mean(axis=1) # [0.75066667 0.70266667 0.71706667 0.70552381 0.71066667]
test_scores_mean = test_scores.mean(axis = 1) # [0.62543736 0.63769059 0.68408761 0.68088463 0.68941519]

plt.figure(figsize=(50,10))
plt.subplot(1,2,1)
plt.grid(True)
plt.plot(train_sizes, train_scores_mean, "y", marker = "o", label = "train")
plt.plot(train_sizes, test_scores_mean, "g", marker = "o", label = "test")
plt.ylim(0.0, 1.05)


rf_classifuire = ensemble.RandomForestClassifier(n_estimators=50, max_depth=10, random_state=3)
train_sizes, train_scores, test_scores = model_selection.learning_curve(rf_classifuire, bioresponse_data, bioresponse_target,
                                                                        train_sizes=np.arange(0.1,1.,0.2),
                                                                        cv = 3, scoring="accuracy")
train_scores_mean  = train_scores.mean(axis=1)
test_scores_mean = test_scores.mean(axis = 1)

plt.subplot(1,2,2)
plt.grid(True)
plt.plot(train_sizes, train_scores_mean, "y", marker = "o", label = "train")
plt.plot(train_sizes, test_scores_mean, "g", marker = "o", label = "test")
plt.ylim(0.0, 1.05)

plt.show()

