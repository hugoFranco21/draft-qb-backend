import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from graphviz import render
from sklearn import tree
import pydot
import pickle

df = pd.read_excel('../datasets/should_you_draft.xlsx', header=1, usecols=[4, 10, 11, 12, 13, 16, 17, 18, 19])
print(df)

df_y = df['Draft']
df_x = df[['W',
    'Cmp%',
    'Yds',
    'TD',
    'Int',
    'Rate',
    'Years',
    'Age']]

X = df_x
y = df_y
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.20, random_state=0)
tree_clf = DecisionTreeClassifier(max_depth = 3)
tree_clf.fit(X_train,y_train)
y_pred_tree= tree_clf.predict(X_test)


dot = tree.export_graphviz(tree_clf, out_file='tree.dot', 
    feature_names=['W',
        'Cmp%',
        'Yds',
        'TD',
        'Int',
        'Rate',
        'Years',
        'Age'],
    class_names=['Yes', 'No'],
    filled=True, rounded=True,  
    special_characters=True) 
(graph,) = pydot.graph_from_dot_file('tree.dot')

graph.write_png('../assets/tree.png')
rnd_clf = RandomForestClassifier(n_estimators=10, max_leaf_nodes=16, n_jobs=-1, random_state=10)
rnd_clf.fit(X_train, y_train)
y_pred_rf= rnd_clf.predict(X_test)
print("random forest", accuracy_score(y_test, y_pred_rf))

f = open("../output/script3.txt", "w")
f.write("Output of the script 3\n")
f.write("\ntree score: " + str(accuracy_score(y_test, y_pred_tree)))
f.write("\nrandom forest score " + str(accuracy_score(y_test, y_pred_rf)))
f.close()

with open('../endpoint/forest.dat', 'wb') as f:
    pickle.dump(rnd_clf, f)


