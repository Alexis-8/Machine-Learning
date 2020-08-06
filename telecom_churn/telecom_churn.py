import pandas as pd
from sklearn import tree
from sklearn import model_selection, metrics
import numpy as np
from sklearn import neighbors
from sklearn.tree import export_graphviz


# Сравнение методов DecisionTreeClassifier и KNeighborsClassifier на данных по оттоку клиентов.

telecom_churn = pd.read_csv("telecom_churn.csv")

# telecom_churn.columns # ['State', 'Account length', 'Area code', 'International plan',
       # 'Voice mail plan', 'Number vmail messages', 'Total day minutes',
       # 'Total day calls', 'Total day charge', 'Total eve minutes',
       # 'Total eve calls', 'Total eve charge', 'Total night minutes',
       # 'Total night calls', 'Total night charge', 'Total intl minutes',
       # 'Total intl calls', 'Total intl charge', 'Customer service calls',
       # 'Churn']

telecom_churn.drop(['State', 'Area code',
       'Voice mail plan', 'Number vmail messages', 'Total day minutes',
       'Total day calls', 'Total day charge', 'Total eve minutes',
       'Total eve calls', 'Total eve charge', 'Total night minutes',
       'Total night calls', 'Total night charge', 'Total intl minutes',
       'Total intl calls', 'Total intl charge', 'Customer service calls'], axis=1, inplace=True)

telecom_churn['International plan'] = telecom_churn['International plan'].map({"Yes": 1,"No": 0 })

target = telecom_churn['Churn'].astype("int")
data = telecom_churn.iloc[:,:-1]

train_data, test_data, train_labels, test_labels = model_selection.train_test_split(data, target, test_size=0.3, random_state=3)

#                                           Decision Tree:

dtc = tree.DecisionTreeClassifier(random_state=17)
cross_score_tree = model_selection.cross_val_score(dtc, train_data, train_labels, cv=5) # [0.85438972 0.84582441 0.85224839 0.8304721  0.84334764]
mean_cross_score_tree = np.mean(cross_score_tree) # 0.845256453851173

# настройка max_depth:

tree_params = {"max_depth": np.arange(1,11),
               "max_features": (0.5,0.7, 0.3, 1)}

dtc_GSCv = model_selection.GridSearchCV(dtc, tree_params, cv=5, n_jobs=-1)
dtc_GSCv.fit(train_data, train_labels)

print(dtc_GSCv.best_score_) # 0.852979937690123
print(dtc_GSCv.best_params_) # {'max_depth': 6, 'max_features': 0.5}


#                                            KNN:

knn = neighbors.KNeighborsClassifier()
cross_score_knn = model_selection.cross_val_score(knn, train_data, train_labels, cv=5)
mean_cross_score_knn = np.mean(cross_score_knn) # [0.84368308 0.83511777 0.83940043 0.8304721  0.83690987]

# настройка knn:

knn_params = {"n_neighbors": list(range(5,30,5)) + list(range(50,100,10))}
knn_GSCv = model_selection.GridSearchCV(knn, knn_params, cv=5)
knn_GSCv.fit(train_data, train_labels)

print(knn_GSCv.best_score_) # 0.8512650375421602    tree is better!
print(knn_GSCv.best_params_)  # {'n_neighbors': 15}


# conclusion:

tree_best_predict = dtc_GSCv.best_estimator_.predict(test_data)
score = dtc_GSCv.score(test_data, test_labels) # 0.857

acc_score = metrics.accuracy_score(test_labels, tree_best_predict) # 0.857
1 - np.mean(target) # 0.8550855085508551 - good client

# draw tree:

export_graphviz(dtc_GSCv.best_estimator_, out_file="telecom_tree.dot", feature_names=telecom_churn.columns(), filled=True)
