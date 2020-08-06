import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('titanic.csv')
data[(data['Embarked'] == 'C') & (data['Fare'] > 20)].sort_values(by = 'Fare', ascending= False)

def age_category(age):
    if age <30:
        return 1
    elif age < 55:
        return 2
    else : return 3

data["Age_category"] = data["Age"].apply(age_category)
data.loc[:10,['Age', 'Age_category']]
data.groupby("Survived")[['Age']].agg(np.mean)
#pd.crosstab(data['Age'],data['Survived'])

d = {'male': 1,'female': 0}
data['Sex'].map(d)

data['Fare'].apply(lambda x: 'rich' if x > 20 else 'pure')
# data['Fare'].plot.hist()

# sns.boxplot(x = data['Fare'])
# data['Survived'].value_counts().head()
# sns.countplot(data['Survived'])
data['Survived'].map({0: 'red', 1:'green'})
# plt.scatter(data["Age"], data['Survived'], color = data['Survived'].map({0: 'red', 1:'green'}))
# sns.heatmap(data.corr())
sns.boxplot(x = 'Age', y = 'Survived', data = data)
plt.show()


