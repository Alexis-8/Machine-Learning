from sklearn import datasets
from pandas import DataFrame
import matplotlib
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns


iris = datasets.load_iris()
iris.keys() # dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
# 'DESCR' - descriptions

# print(iris.data[:10])
# print(iris.target)

iris_frame = DataFrame(iris.data)
iris_frame.columns = iris.feature_names
iris_frame["target"] = iris.target

iris_frame["target"] = iris_frame["target"].apply(lambda x: iris["target_names"][x])

iris_frame[iris_frame["target"] == "setosa"].hist() # histogram for each features for setosa

# all features for all data:

plt.figure(figsize=(20,24))
plot_number = 0
for feature_name in iris['feature_names']:
    for target_name in iris["target_names"]:
        plot_number += 1
        plt.subplot(4,3,plot_number)
        plt.hist(iris_frame[iris_frame["target"] == target_name][feature_name])

        plt.title(target_name)
        plt.xlabel("cm")
        plt.ylabel(feature_name[:-4])

# pair of features:

sns.set(style="ticks", palette="husl")
sns.pairplot(iris_frame, hue="target")

# # another way:

# sns.set(font_scale=1.3)
# data = sns.load_dataset("iris")
# sns.pairplot(data,hue="species")

plt.show()