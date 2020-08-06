from matplotlib.colors import ListedColormap
from sklearn import model_selection
from sklearn import tree, metrics
from sklearn import datasets
import matplotlib.pyplot as plt

import numpy as np

classification_problem = datasets.make_classification(n_features=2, n_informative=2, n_classes=3, n_redundant=0,
                                                      n_clusters_per_class=1, random_state=5)
colors = ListedColormap(['red', 'blue', 'yellow'])
light_colors = ListedColormap(["lightcoral", "thistle", "lightyellow"])
plt.figure(figsize=(8,6))
# print(classification_problem)
plt.scatter(list(map(lambda x: x[0], classification_problem[0])), list(map(lambda x: x[1],classification_problem[0])),
            c = classification_problem[1], cmap=colors, s=100)

train_data, test_data, train_labels, test_labels = model_selection.train_test_split(classification_problem[0],
                                                                                    classification_problem[1],
                                                                                    test_size=0.3, random_state=2)
clf = tree.DecisionTreeClassifier(random_state=2)
clf.fit(train_data, train_labels)

predictions = clf.predict(test_data)
acc_score = metrics.accuracy_score(test_labels, predictions) # 0.7333333333333333

# decision surface:

def get_meshgrid(data, step = .05, border = .5,):
    x_min, x_max = data[:,0].min() - border, data[:,0].max() + border
    y_min, y_max = data[:,0].min() - border, data[:,0].max() + border
    return np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

def plot_decision_surface(estimator, train_data, train_labels, test_data, test_labels, colors = colors, light_colors = light_colors):
    estimator.fit(train_data, train_labels)
    plt.figure(figsize=(16,6))

    # plot on train data:
    plt.subplot(1,2,1)
    xx,yy = get_meshgrid(train_data)
    mesh_predictions = np.array(estimator.predict(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
    plt.pcolormesh(xx,yy, mesh_predictions, cmap = light_colors)
    plt.scatter(train_data[:,0], train_data[:,1], c = train_labels, s = 100, cmap=colors)
    plt.title("Train data, accuracy = {:.2f}".format(metrics.accuracy_score(train_labels, estimator.predict(train_data))))

    # plot on test data:
    plt.subplot(1,2,2)
    plt.pcolormesh(xx,yy, mesh_predictions, cmap=light_colors)
    plt.scatter(test_data[:,0], test_data[:,1], c = test_labels, s=100, cmap=colors)
    plt.title("Test data, accuracy = {:.2}".format(metrics.accuracy_score(test_labels, estimator.predict(test_data))))

# depth = 1:

estimator1 = tree.DecisionTreeClassifier(random_state=5, max_depth=1)
plot_decision_surface(estimator1, train_data, train_labels, test_data, test_labels)

# depth = 2:

estimator2 = tree.DecisionTreeClassifier(random_state=5, max_depth=2)
plot_decision_surface(estimator2, train_data, train_labels, test_data, test_labels)

# depth = 3:
estimator3 = tree.DecisionTreeClassifier(random_state=5, max_depth=3)
plot_decision_surface(estimator3, train_data, train_labels, test_data, test_labels)

# no restricted depth: --- overfitting

plot_decision_surface(tree.DecisionTreeClassifier(random_state=5), train_data, train_labels, test_data, test_labels)

# with min_sample_leaf:

plot_decision_surface(tree.DecisionTreeClassifier(random_state=5, min_samples_leaf=3), train_data, train_labels, test_data, test_labels)


plt.show()
