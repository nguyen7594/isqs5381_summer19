# -*- coding: utf-8 -*-
"""
Created on Sun May 19 09:37:23 2019

@author: Nguyen7594
"""


# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common libraries
import numpy as np
import pandas as pd
import os

np.random.seed(42)

# To plot pretty figures
%matplotlib inline 
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes',labelsize=7)
mpl.rc('xtick',labelsize=6)
mpl.rc('ytick',labelsize=6)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
    
    
def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]    

# Import MNIST    
try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    mnist.target = mnist.target.astype(np.int8) # fetch_openml() returns targets as strings
    sort_by_target(mnist) # fetch_openml() returns an unsorted dataset
except ImportError:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')
mnist["data"], mnist["target"]    

X, y = mnist["data"], mnist["target"]
X.shape
y.shape

# Try some images
some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = mpl.cm.binary,
           interpolation="nearest")
plt.axis("off")
plt.show()

# Split test-train datasets
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

## Binary classifier ##
# Stochastic Gradient Descent (SGD)
from sklearn.linear_model import SGDClassifier
# eg: if it is 5 or not
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train,y_train_5)


## Performance measures ##
# Measure accuracy using cross-validation
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf,X_train,y_train_5,cv=3,scoring='accuracy')

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3,random_state=42)

for train_index, test_index in skfolds.split(X_train,y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_folds = X_train[test_index]
    y_test_folds = (y_train_5[test_index])
    
    clone_clf.fit(X_train_folds,y_train_folds)
    y_pred = clone_clf.predict(X_test_folds)
    n_correct = sum(y_pred == y_test_folds)
    print(n_correct/len(y_test_folds))

from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")


#Confusion matrix
from sklearn.model_selection import cross_val_predict 
y_train_pred = cross_val_predict(sgd_clf,X_train,y_train_5,cv=3)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5,y_train_pred)
y_train_perfect_predictions = y_train_5
confusion_matrix(y_train_5, y_train_perfect_predictions)

#Precision and Recall
from sklearn.metrics import precision_score,recall_score
precision_score(y_train_5,y_train_pred)
recall_score(y_train_5,y_train_pred)

#F1 score
from sklearn.metrics import f1_score
f1_score(y_train_5,y_train_pred)

#Precision/Recall trade off
y_scores = sgd_clf.decision_function([some_digit])
y_scores
threshold=0
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred
threshold=200000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred

y_scores = cross_val_predict(sgd_clf,X_train,y_train_5,cv=3,method='decision_function')

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5,y_scores) 

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

# review python
a = [1,2,3,4,5,6,7,8,9,10]
b=[]
for i in range(len(a)):
    b.append(a[-(i+1)])
print(b)
a.reverse()
a[::-1]
a= 'abcde'
a[-1]
len(a)
#---------------------------------#
y_train_pred_90 = (y_scores > 70000)
precision_score(y_train_5,y_train_pred_90)
recall_score(y_train_5,y_train_pred_90)


#The ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds= roc_curve(y_train_5,y_scores)
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
plot_roc_curve(fpr, tpr)
plt.show()
#AUC
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5,y_scores)

#RandomForest
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf,X_train,y_train_5,cv=3,
                                    method='predict_proba')
y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)
plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="bottom right")
plt.show()
roc_auc_score(y_train_5, y_scores_forest)
#precision v. recall
y_probas_forest = cross_val_predict(forest_clf,X_train,y_train_5,cv=3)
precision_score(y_train_5,y_probas_forest)
recall_score(y_train_5,y_probas_forest)


## Multiclass classification
sgd_clf.fit(X_train,y_train)
sgd_clf.predict([some_digit])
sgd_clf.decision_function([some_digit])
sgd_clf.classes_
#OnevsOneClassifier
from sklearn.multiclass import OneVsOneClassifier
ovo_clf=OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train,y_train)
ovo_clf.predict([some_digit])
len(ovo_clf.estimators_)
#RandomForestClassifier
forest_clf.fit(X_train,y_train) 
forest_clf.predict([some_digit])
forest_clf.predict_proba([some_digit])
#Evaluate
cross_val_score(sgd_clf,X_train,y_train,cv=3,scoring='accuracy')
#scaling the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf,X_train_scaled,y_train,cv=3,scoring='accuracy')


## Error Analysis
y_train_pred = cross_val_predict(sgd_clf,X_train_scaled,y_train,cv=3)
conf_mx = confusion_matrix(y_train,y_train_pred) 
conf_mx
plt.matshow(conf_mx,cmap=plt.cm.gray)
plt.show()
# plot error
row_sums = conf_mx.sum(axis=1,keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx,0)
plt.matshow(norm_conf_mx,cmap=plt.cm.gray)
plt.show()



## Multilabel classification
from sklearn.neighbors import KNeighborsClassifier 
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
knn_clf.predict([some_digit])
#evaluate
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)

## Multioutput classification




### Exercise 3 ###
# Common libraries
import numpy as np
import pandas as pd
import os
import seaborn as sns

np.random.seed(42)

# To plot pretty figures
%matplotlib inline 
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes',labelsize=7)
mpl.rc('xtick',labelsize=6)
mpl.rc('ytick',labelsize=6)

# read current dir
cr = os.getcwd()

# get file
TITANIC_PATH = os.path.join("Documents","Python","Hands_on_ML")
def load_titanic_data(file_name,titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path,file_name)
    return pd.read_csv(csv_path)
#train file
train_data = load_titanic_data("train_titanic.csv")
train_data.head()
#test file
test_data = load_titanic_data("test_titanic.csv")
test_data.head()


#Overview dataset
train_data.info()
y_train = train_data['Survived'].copy()
num_vab = ['Age','SibSp','Parch','Fare']
cat_vab = ['Pclass','Sex','Embarked']
## 891 variables, missing value: age, embarked
train_data['Embarked'].value_counts().reset_index().iloc[0,0] 
train_data['Sex'].value_counts() 
train_data['Pclass'].value_counts() 
train_data[num_vab].hist(bins=50)
corr_matrix = train_data.corr()
corr_matrix['Survived']
sns.heatmap(corr_matrix)
sns.catplot(x='Survived',y='Fare',data=train_data,kind='box')
sns.countplot(x='Pclass',hue='Survived',data=train_data)
sns.catplot(x='Sex',y='Age',hue='Survived',data=train_data,kind='box')


## Data cleaning ##
# custom transformer
from sklearn.base import BaseEstimator, TransformerMixin
# select columns
class dfselector(BaseEstimator, TransformerMixin):
    def __init__(self,feature_names):
        self._feature_names = feature_names
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        return X[self._feature_names]
    
  
# fill missing value for all categories by most popular category
class fillna_cat(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        for i in list(X.columns):
            X.loc[:,i] = X[i].fillna(train_data[i].value_counts().reset_index().iloc[0,0])
        return X 

# test above class  
t = fillna_cat()    
test = pd.DataFrame(t.fit_transform(train_data),columns=cat_vab)     
test.info()  
test['Embarked'].value_counts()
train_data['Embarked'].value_counts()

# best ver for filling missing value for categorical variables
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)    
# note    
for c in train_data:
    print(c)       
list(train_data.columns)    
    
# num pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
num_pipeline = Pipeline([('select_num',dfselector(num_vab)),
                         ('imputer',SimpleImputer(strategy='median')),
                         ('std_scaler',StandardScaler())])         
# cat pipeline
from sklearn.preprocessing import OneHotEncoder    
cat_pipeline = Pipeline([('select_cat',dfselector(cat_vab)),
                         ('imputer',MostFrequentImputer()),
                         ('encoder',OneHotEncoder(sparse=False))])    
    

# combine num and cat pipelines
from sklearn.pipeline import FeatureUnion    
preprocess_pipeline=FeatureUnion(transformer_list=[
        ('num_pipeline',num_pipeline),
        ('cat_pipeline',cat_pipeline)])
x_train = preprocess_pipeline.fit_transform(train_data)     
y_train.head()


# Model Selection & Train
    