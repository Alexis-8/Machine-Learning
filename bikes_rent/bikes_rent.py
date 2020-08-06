import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model, preprocessing, metrics, ensemble


# UCI - help

data = pd.read_csv("bikes_rent.csv")

# visual analisis:

plt.rcParams["figure.figsize"] = (12, 8)
sns.violinplot(data["season"], data["cnt"])

plt.figure()
sns.heatmap(data.corr())

plt.figure()
for i,col in enumerate(data.columns[:-1]):
    plt.subplot(4,3,i+1)
    plt.scatter(data[col], data["cnt"])
    plt.title(col)

# plt.figure()
# plt.scatter(data["mnth"], data["cnt"])
# plt.scatter(data["temp"], data["cnt"])

plt.figure()
data["cnt"].hist()
plt.show()

# Models:

linreg = linear_model.LinearRegression()
lasso = linear_model.Lasso(random_state=3)
ridge = linear_model.Ridge(random_state=3)
lasso_cv = linear_model.LassoCV(random_state=3)
ridge_cv = linear_model.RidgeCV()

forest = ensemble.RandomForestRegressor(random_state=3, n_estimators=500)

# Data:

X, Y = data.drop("cnt", axis=1).values, data["cnt"].values

train_part_size = int(.7 * X.shape[0]) # 70% train

X_train, X_valid = X[:train_part_size, :], X[train_part_size:,:]
Y_train, Y_valid = Y[:train_part_size], Y[train_part_size:]

# Scale:

scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Fitting:

linreg.fit(X_train_scaled, Y_train)

# MSE:

MSE = metrics.mean_squared_error(Y_valid, linreg.predict(X_valid_scaled))
error_count_bike = np.sqrt(MSE) # 1120.7214094932299

coef_linreg = pd.DataFrame(linreg.coef_, data.columns[:-1], columns=["coef"]).sort_values(by = "coef", ascending=False)

print(coef_linreg) # multicolinear


def train_validate_report(model, X_train_scaled, Y_train, X_valid_scaled, Y_valid, feature_names, forest = False):
    """ Only for linear models and regression trees """
    model.fit(X_train_scaled, Y_train)
    print("MSE: ", np.sqrt(metrics.mean_squared_error(Y_valid,
                                             model.predict(X_valid_scaled))))

    print("Model coeff:")
    coef = model.feature_importances_ if forest else model.coef_
    coef_name = "Importance" if forest else "Coef"
    print(pd.DataFrame(coef, feature_names,
                       columns=[coef_name]).sort_values(by = coef_name, ascending=False))

# Different models compare:

train_validate_report(lasso, X_train_scaled, Y_train, X_valid_scaled, Y_valid, feature_names=data.columns[:-1]) # MSE:  1120.7214094932299
train_validate_report(lasso_cv, X_train_scaled, Y_train, X_valid_scaled, Y_valid, feature_names=data.columns[:-1]) # MSE:  1120.7436606195301
train_validate_report(ridge, X_train_scaled, Y_train, X_valid_scaled, Y_valid, feature_names=data.columns[:-1]) # MSE:  1119.5605202108343
train_validate_report(ridge_cv, X_train_scaled, Y_train, X_valid_scaled, Y_valid, feature_names=data.columns[:-1]) # MSE:  1118.9035433219844

train_validate_report(forest, X_train, Y_train, X_valid, Y_valid, feature_names=data.columns[:-1], forest=True) # MSE:  1048.1909454899992

