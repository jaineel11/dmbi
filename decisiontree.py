import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

data = {'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 3, 4, 5, 6],
        'target': ['A', 'B', 'A', 'B', 'A']}
df = pd.DataFrame(data)

X = df[['feature1', 'feature2']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

plt.figure(figsize=(25, 5))
tree.plot_tree(clf, filled=True, feature_names=X.columns.tolist(), class_names=y.unique().tolist())
plt.show()