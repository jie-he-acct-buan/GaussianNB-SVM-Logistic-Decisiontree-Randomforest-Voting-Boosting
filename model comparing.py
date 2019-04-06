import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

###############################################################################
df = pd.read_csv('meningitis.csv', ' ', index_col='ID')
X = df.drop(['MENINGITIS'], axis = 1).values
y = df.MENINGITIS.values

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, y)

X_predict = np.array([[True, True, True]])
print('Does a patient with headache, fever, and vomiting have meningitis?', '\n', clf.predict(X_predict))
print('with the probability of ', clf.predict_proba(X_predict)[0,0])

###############################################################################
voice = pd.read_csv('voice.csv')

print(voice.head())

from sklearn.model_selection import train_test_split
X = voice.iloc[:,0:19]
y = voice.iloc[:,20]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print(y.describe())


# Train a SVM model; tune the best parameters C and gamma; 
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svc = SVC(kernel = 'rbf', probability=True, random_state=42)
params_svc = {'C':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
              'gamma':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 10000]}

grid_svc = GridSearchCV(svc, params_svc, cv=5)
grid_svc.fit(X_train, y_train)

y_pred = grid_svc.predict(X_test)
print(grid_svc.best_params_)
print('SVC accuracy score: ', accuracy_score(y_test, y_pred))


# Train a logistic regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver="liblinear", random_state=42)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('logreg accuracy score: ', accuracy_score(y_test, y_pred))


# Train a decision tree (you need to select proper parameters) 
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=42)
param_DT = {"max_depth": range(1,10),
           "min_samples_split": range(2,20),
           "max_leaf_nodes": range(2,20)}
grid_tree = GridSearchCV(tree, param_DT, cv=5)
grid_tree.fit(X_train,y_train)
y_pred = grid_tree.predict(X_test)
print(grid_tree.best_params_)
print('decision tree accuracy score: ', accuracy_score(y_test, y_pred))


# Train a random forest model
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators = 500, random_state = 42)
param_RF = {'max_leaf_nodes':range(2,20)}
grid_RF = GridSearchCV(RF, param_RF, cv=5)
grid_RF.fit(X_train, y_train)
y_pred = grid_RF.predict(X_test)
print(grid_RF.best_params_)
print('random forest accuracy score: ', accuracy_score(y_test, y_pred))


# use voting classifier to combine the above four model
from sklearn.ensemble import VotingClassifier
svc = SVC(kernel = 'rbf', C=10000, gamma=0.001, random_state=42)
logreg = LogisticRegression(solver="liblinear", random_state=42)
tree = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=15, min_samples_split=2, random_state=42)
RF = RandomForestClassifier(n_estimators=500, max_leaf_nodes=17, random_state=42)
voting = VotingClassifier(estimators=[('svc', svc), ('logreg', logreg), ('tree', tree), ('RF', RF)], voting='soft')
voting.fit(X_train, y_train)

for clf in (svc, logreg, tree, RF, voting):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


# apply decision tree with adaboost
from sklearn.ensemble import AdaBoostClassifier

ada_tree = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5, max_leaf_nodes=15, min_samples_split=2, random_state=42), 
                           n_estimators=500, algorithm='SAMME.R', learning_rate=0.5, random_state=42)
ada_tree.fit(X_train, y_train)
y_pred = ada_tree.predict(X_test)
print('adaboost decision tree accuracy score: ', accuracy_score(y_test, y_pred))


# apply svm with adaboost
ada_svm = AdaBoostClassifier(SVC(kernel = 'rbf', C=10000, gamma=0.001, random_state=42),
                             n_estimators=500, algorithm='SAMME', learning_rate=0.5, random_state=42)
ada_svm.fit(X_train, y_train)
y_pred = ada_svm.predict(X_test)
print('adaboost svm accuracy score: ', accuracy_score(y_test, y_pred))


# Plot the roc curves of the above NINE models
svc = SVC(kernel = 'rbf', C=10000, gamma=0.001, probability=True, random_state=42)
logreg = LogisticRegression(solver="liblinear", random_state=42)
tree = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=15, min_samples_split=2, random_state=42)
RF = RandomForestClassifier(n_estimators=500, max_leaf_nodes=17, random_state=42)
voting = VotingClassifier(estimators=[('svc', svc), ('logreg', logreg), ('tree', tree), ('RF', RF)], voting='soft')
ada_tree = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5, max_leaf_nodes=15, min_samples_split=2, random_state=42), 
                           n_estimators=100, algorithm='SAMME.R', learning_rate=0.5, random_state=42)
ada_svm = AdaBoostClassifier(SVC(kernel = 'rbf', C=10000, gamma=0.001, probability=True, random_state=42),
                             n_estimators=100, algorithm='SAMME', learning_rate=0.5, random_state=42)

clfs = {'SVM':svc, 'Logistic':logreg, 'Decision Tree':tree, 'Random Forest':RF, 'Voting':voting, 
        'AdaBoost Decision Tree':ada_tree, 'AdaBoost SVM':ada_svm}

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

for key, value in clfs.items():
    value.fit(X_train, y_train)
    probs = value.predict_proba(X_test)[:,1]
    fpr, tpr, threshold = roc_curve(y_test, probs, pos_label='male')
    roc_auc = roc_auc_score(y_test, probs)
    plt.figure(num=None, figsize=(4,4), dpi=80, facecolor='w', edgecolor='k')
    plt.title(key + ' ROC')
    plt.plot(fpr, tpr, 'b', label = 'AUC = ' + str(roc_auc))
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()







