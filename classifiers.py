from pyspark.sql import *
import helper_function
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from sklearn import metrics
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import pandas as pd
start_time = time.time()


path = 'D:/2020spring/6230/project/FINAL/'
filename = 'all.train'
###laod and split svmlight file
X, y= helper_function.get_data(path, filename)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_female, y_female = helper_function.get_data('D:/2020spring/6230/project/FINAL/', "female.test")
X_train_female, X_test_female, y_train_female, y_test_female = train_test_split(X_female, y_female, test_size=0.29, random_state=42)

X_male, y_male = helper_function.get_data('D:/2020spring/6230/project/FINAL/', "male_new.test")
X_train_male, X_test_male, y_train_male, y_test_male = train_test_split(X_male, y_male, test_size=0.23, random_state=42)


# Instantiate the classfiers and make a list
classifiers = [
    KNeighborsClassifier(3),
    LogisticRegression(random_state=42),
    DecisionTreeClassifier(random_state=42),
    RandomForestClassifier(random_state=42),
    AdaBoostClassifier(random_state=42),
    GradientBoostingClassifier(random_state=42)]

# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])
# Train the models and record the results
for cls in classifiers:
    model = cls.fit(X_train, y_train)
    yproba = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  yproba)
    auc = metrics.roc_auc_score(y_test, yproba)
    name = cls.__class__.__name__
    result_table = result_table.append({'classifiers':name,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)
result_table.to_csv("auc_all.csv", index=False)
######try one model###############################################################
#cls = classifiers[1]
#model = cls.fit(X_train, y_train)
#yproba = model.predict_proba(X_test_male)[:,1]
#fpr, tpr, _ = metrics.roc_curve(y_test_male,  yproba)
#auc = metrics.roc_auc_score(y_test_male, yproba)
#name = cls.__class__.__name__
#result_table = result_table.append({'classifiers':name,
#                                    'fpr':fpr, 
#                                    'tpr':tpr, 
#                                    'auc':auc}, ignore_index=True)
################################################################################

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)
##plot 
fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')
import numpy as np
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis_male Patients', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()
fig.savefig('multiple_roc_curve_male.png')






















