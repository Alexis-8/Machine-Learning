from sklearn import linear_model, model_selection, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

raw_data = pd.read_csv("train.csv", header = 0, sep = ",")
raw_data.isnull().values.any() # False   if where is null values
raw_data["datetime"] = raw_data["datetime"].apply(pd.to_datetime)

raw_data["month"] = raw_data["datetime"].apply(lambda x: x.month)
raw_data["hour"] = raw_data["datetime"].apply(lambda x: x.hour)

train_data = raw_data.iloc[:-1000, :]
hold_dot_test_data = raw_data.iloc[-1000:,:]

print("train from {} to {}".format(train_data.datetime.min(),train_data.datetime.max()))
print("test from {} to {}".format(hold_dot_test_data.datetime.min(),hold_dot_test_data.datetime.max()))

# training:

train_labels = train_data["count"].values
train_data = train_data.drop(["datetime", "count"], axis = 1)

# test:

test_labels = hold_dot_test_data["count"].values
test_data = hold_dot_test_data.drop(["datetime", "count"], axis = 1)

# visualisation:

plt.figure(figsize= (20,10))

plt.subplot(1,2,1)
plt.hist(train_labels)
plt.title("train data")

plt.subplot(1,2,2)
plt.hist(test_labels)
plt.title("test data")



numeric_features= ["temp", 'atemp', 'humidity', 'windspeed', 'casual', 'registered','month', 'hour']

train_data = train_data[numeric_features]
test_data = test_data[numeric_features]

regressor = linear_model.SGDRegressor(random_state=0)
regressor.fit(train_data,train_labels)

metrics1 = metrics.mean_absolute_error(test_labels, regressor.predict(test_data)) # very big number 20997892275395.24

# scaling

scaler = StandardScaler()
scaler.fit(train_data, train_labels)

scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)

regressor.fit(scaled_train_data, train_labels)
metrics2 = metrics.mean_absolute_error(test_labels, regressor.predict(scaled_test_data)) # too good 0.04293048301240903

reg_coef = list(map(lambda x: round(x,2), regressor.coef_))  # [0.46, -0.45, 0.0, -0.01, 50.86, 148.01, -0.0, 0.01]
# bad feautures: 50, 148
train_data.drop(['casual', 'registered'], axis = 1, inplace = True)
test_data.drop(['casual', 'registered'], axis = 1, inplace = True)

scaler.fit(train_data,train_labels)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)

regressor.fit(scaled_train_data, train_labels)
metrics3 = metrics.mean_absolute_error(test_labels,regressor.predict(scaled_test_data))

reg_coef2 = list(map(lambda x: round(x,2), regressor.coef_)) # [30.01, 32.15, -42.28, 3.78, 12.71, 50.06]

# PipeLine:

pipeline = Pipeline(steps = [("scaling", scaler), ("regression", regressor)])
pipeline.fit(train_data,train_labels)
metrics4 = metrics.mean_absolute_error(test_labels, pipeline.predict(test_data)) # 121.8835371361759

# Selections parameters:

pipeline.get_params().keys() # dict_keys(['memory', 'steps', 'verbose', 'scaling', 'regression', 'scaling__copy', 'scaling__with_mean', 'scaling__with_std', 'regression__alpha', 'regression__average', 'regression__early_stopping', 'regression__epsilon', 'regression__eta0', 'regression__fit_intercept', 'regression__l1_ratio', 'regression__learning_rate', 'regression__loss', 'regression__max_iter', 'regression__n_iter_no_change', 'regression__penalty', 'regression__power_t', 'regression__random_state', 'regression__shuffle', 'regression__tol', 'regression__validation_fraction', 'regression__verbose', 'regression__warm_start'])
parameters_grid = {'regression__alpha':[0.0001, 0.01],
'regression__loss': ['huber', "squared_loss", "epsilon_insensitive"],
'regression__n_iter_no_change': [3,5,10,50],
                   'regression__penalty': ['l1','l2','none'],
'scaling__with_mean': [0,1] }

grid_cv = model_selection.GridSearchCV(pipeline, parameters_grid, scoring="neg_mean_absolute_error", cv = 4)
grid_cv.fit(train_data, train_labels)

print(grid_cv.best_score_) # -110.68217235388403
print(grid_cv.best_params_) # {'regression__alpha': 0.01, 'regression__loss': 'epsilon_insensitive', 'regression__n_iter_no_change': 3, 'regression__penalty': 'l1', 'scaling__with_mean': 0}

metrics5 = metrics.mean_absolute_error(test_labels,grid_cv.best_estimator_.predict(test_data))
mean = np.mean(test_labels)

test_predictions = grid_cv.best_estimator_.predict(test_data)

print(test_predictions[:10])

plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.grid(True)
plt.scatter(train_labels,pipeline.predict(train_data), alpha=0.5, color = "red")
plt.scatter(test_labels, pipeline.predict(test_data), alpha=0.5, color = "blue")
plt.title("no parameters setting")
plt.xlim(-100,100)
plt.ylim(-100,100)


plt.subplot(1,2,2)
plt.grid(True)
plt.scatter(train_labels, grid_cv.best_estimator_.predict(train_data), alpha=0.5, color = "red")
plt.scatter(test_labels,grid_cv.best_estimator_.predict(test_data), alpha=0.5, color = "blue")
plt.title("grid search")
plt.xlim(-100,100)
plt.ylim(-100,100)

plt.show()