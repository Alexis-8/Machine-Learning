from sklearn import datasets
import matplotlib
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# plotting:

def plot_2d_dataset(data,colors):
    fig,ax = plt.subplots(figsize=(8,8))
    ax.scatter(list(map(lambda x: x[0], data[0])), list(map(lambda x: x[1], data[0])),
                   c = data[1], cmap=colors)

# easy dataset:

circles = datasets.make_circles()
# print("features: {}".format(circles[0][:10]))
# print("target: {}".format(circles[1][:10])) # metki classov
colors = ListedColormap(["red", "blue"])

noisy_circles = datasets.make_circles(noise=0.05) # impact on the complexity of the task

plot_2d_dataset(noisy_circles,colors)

# random classification problem:

simple_classification_problem = datasets.make_classification(n_features=4, n_informative=4, n_classes=4,
                                                             n_redundant=0, n_clusters_per_class=1,
                                                             random_state=1)

colors1 = ListedColormap(["red", "yellow","green","blue"])

plot_2d_dataset(simple_classification_problem,colors1)

plt.show()

