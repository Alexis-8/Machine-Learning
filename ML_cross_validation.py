import sklearn
from sklearn import model_selection
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

# one-time splitting:

train_data, test_data, train_lables, test_labels = model_selection.train_test_split(iris.data, iris.target,
                                                                                    test_size=0.3) #30% test
len(test_labels)/len(iris.data) # 0.3

# Strategy of cross-validation:

#   1. KFold:

kf = model_selection.KFold(n_splits= 3)
X =  np.array([[1 ,  2 ],  [ 3 ,  4 ],  [ 1 ,  2 ],  [ 3 ,  4 ]])
for train_indeces, test_indeces in kf.split(X):
    print(train_indeces, test_indeces)

# 2. StratifieldKFold:

XS =  np.array([[1 ,  2 ],  [ 3 ,  4 ],  [ 1 ,  2 ],  [ 3 ,  4 ]])
target = np.array([0]*2+[1]*2)
print(target)
skf = model_selection.StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
for train_indeces, test_indeces in skf.split(XS,target):
    print(train_indeces, test_indeces)

# 3. ShaffleSplit:

shst = model_selection.ShuffleSplit(n_splits=10, test_size=0.2)
for train_indeces, test_indeces in shst.split(XS, target):
    print("ShaffleSplit", train_indeces, test_indeces)

# 4. StratifiedShuffleSplit:

Sshst = model_selection.StratifiedShuffleSplit(n_splits=4, test_size=0.5)
for train_indeces, test_indeces in Sshst.split(XS,target):
    print("StratifiedShuffleSplit", train_indeces, test_indeces)

# 5. LeaveOneOut:

loo = model_selection.LeaveOneOut()
for train_indeces, test_indeces in loo.split(XS):
    print("LeaveOneOut", train_indeces, test_indeces)