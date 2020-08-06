from sklearn import linear_model, model_selection, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import pipeline, preprocessing
from sklearn.ensemble import RandomForestRegressor

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

raw_data = pd.read_csv("train.csv", header = 0, sep = ",")

raw_data["datetime"] = raw_data["datetime"].apply(pd.to_datetime)

raw_data["month"] = raw_data["datetime"].apply(lambda x: x.month)
raw_data["hour"] = raw_data["datetime"].apply(lambda x: x.hour)

train_data = raw_data.iloc[:-1000, :]
hold_dot_test_data = raw_data.iloc[-1000:,:]

# training:

train_labels = train_data["count"].values
train_data = train_data.drop(["datetime", "count"], axis = 1)

# test:

test_labels = hold_dot_test_data["count"].values
test_data = hold_dot_test_data.drop(["datetime", "count"], axis = 1)

binary_data_columns = ['holiday','workingday']
binary_data_indeces = np.array([(column in binary_data_columns) for column in train_data.columns], dtype = bool)
# [False  True  True False False False False False False False False False]

categorical_data_columns = ["season", "weather", 'month']
categorical_data_indeces = np.array([(column in categorical_data_columns) for column in train_data.columns], dtype = bool)
# [ True False False  True False False False False False False  True False]

numeric_data_columns = ["temp", 'atemp', 'humidity', 'windspeed','hour']
numeric_data_indeces = np.array([(column in numeric_data_columns) for column in train_data.columns], dtype = bool)

# Pipeline:

regressor = linear_model.SGDRegressor(random_state=0, n_iter_no_change = 3, loss="squared_loss", penalty="l1")

estimator = pipeline.Pipeline(steps=[
    ("feature_processing", pipeline.FeatureUnion(transformer_list = [
        # binary:
        ("binary_variables_processing", preprocessing.FunctionTransformer(lambda data: data.iloc[:,binary_data_indeces])),
        # numeric:
        ("numeric_variables_processing", pipeline.Pipeline(steps = [
            ("selecting", preprocessing.FunctionTransformer(lambda data: data.iloc[:, numeric_data_indeces])),
            ("scaling", preprocessing.StandardScaler(with_mean=0.))
        ])),
        # categorical:
        ("categorical_variables_processing", pipeline.Pipeline(steps=[
            ("selecting", preprocessing.FunctionTransformer(lambda data: data.iloc[:, categorical_data_indeces])),
            ("hot_encoding", preprocessing.OneHotEncoder(handle_unknown="ignore"))
        ])),

    ])), ("model_fitting", regressor)])

estimator.fit(train_data, train_labels)
metrics_MAE = metrics.mean_absolute_error(test_labels,estimator.predict(test_data)) #  124.668187966137

parameters_grid = {"model_fitting__alpha": [0.0001, 0.001, 0.1],
                   "model_fitting__eta0": [0.001, 0.05]}
grid_cv = model_selection.GridSearchCV(estimator, parameters_grid, scoring="neg_mean_absolute_error", cv = 4)
grid_cv.fit(train_data, train_labels)
grid_cv.best_score_ # -116.84846662410588
grid_cv.best_params_ # {'model_fitting__alpha': 0.1, 'model_fitting__eta0': 0.001}

test_predictions = grid_cv.best_estimator_.predict(test_data)
MAE = metrics.mean_absolute_error(test_labels,test_predictions) # 121.49718465149938  bad predictions

plt.figure(figsize=(8,6))
plt.grid(True)
plt.xlim(-100,1100)
plt.ylim(-100, 1100)
plt.scatter(train_labels, grid_cv.best_estimator_.predict(train_data), alpha=0.1, color = "red")
plt.scatter(test_labels, grid_cv.best_estimator_.predict(test_data), alpha=0.1, color = "blue")

regressor3 = RandomForestRegressor(random_state=0, max_depth=20, n_estimators=50)
estimator3 =  pipeline.Pipeline(steps=[
    ("feature_processing", pipeline.FeatureUnion(transformer_list = [
        # binary:
        ("binary_variables_processing", preprocessing.FunctionTransformer(lambda data: data.iloc[:,binary_data_indeces])),
        # numeric:
        ("numeric_variables_processing", pipeline.Pipeline(steps = [
            ("selecting", preprocessing.FunctionTransformer(lambda data: data.iloc[:, numeric_data_indeces])),
            ("scaling", preprocessing.StandardScaler(with_mean=0.))
        ])),
        # categorical:
        ("categorical_variables_processing", pipeline.Pipeline(steps=[
            ("selecting", preprocessing.FunctionTransformer(lambda data: data.iloc[:, categorical_data_indeces])),
            ("hot_encoding", preprocessing.OneHotEncoder(handle_unknown="ignore"))
        ])),

    ])), ("model_fitting", regressor3)])
estimator3.fit(train_data, train_labels)
MEA3 = metrics.mean_absolute_error(test_labels, estimator3.predict(test_data))

plt.figure(figsize=(16,6))
plt.subplot(1,2,2)
plt.grid(True)
plt.xlim(-100,1100)
plt.ylim(-100,1100)
plt.scatter(train_labels, estimator3.predict(train_data), alpha=0.5, color ='red')
plt.scatter(test_labels, estimator3.predict(test_data), alpha=0.5, color ='blue')
plt.title("random_forest_model")

plt.show()
