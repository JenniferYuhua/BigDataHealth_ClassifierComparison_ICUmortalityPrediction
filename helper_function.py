from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
import pandas as pd
import numpy as np

def readCSV(file):
    path = "../data/"
    from pyspark.context import SparkContext
    from pyspark.sql.session import SparkSession
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)
    df = spark.read.csv(path + file, inferSchema=True, header=True)
    return df

def id_map(events):
    event_id_all = events.EVENT_ID.unique()
    indx = np.arange(len(event_id_all))
    data = {'EVENT_ID':sorted(event_id_all), 'ID':indx}
    feature_map = pd.DataFrame(data, columns=['EVENT_ID','ID'])
    return feature_map

def map_id_event_id(file, id_map_file):
    file['EVENT_ID'] = file['EVENT_ID'].map(id_map_file.set_index('EVENT_ID')['ID'])
    return file



def patient_feature(event_all):
    patient_features = {}
    for row in event_all.itertuples(index=False):
        if row[0] not in patient_features:
            patient_features[row[0]] = [(row[1], row[2])]
        else:
            patient_features[row[0]].append((row[1],row[2]))
    return patient_features
    
def patient_flag_dic(patients):
    patient_flag = {}
    for row in patients.itertuples(index=False):
        patient_flag[row[0]] = row[2]
    return patient_flag
    
def svmlight(path, filename, patient_features, patient_flag):
    svm_file = path + filename
    deliverable = open(svm_file, 'wb')
    line_svm = ''
    for key,value in sorted(patient_features.items()):
        line_svm += str(patient_flag[key]) + ' '
        for value in sorted(value):
            line_svm += str(value[0]) + ':' + str(value[1]) + ' '
        line_svm += '\n'                    
    deliverable.write(bytes((line_svm),'UTF-8')); #Use 'UTF-8'
    deliverable.close()
    
def get_data(path, filename):
    svmfile = path + filename
    from sklearn.datasets import load_svmlight_file
    data = load_svmlight_file(svmfile)
    return data[0], data[1]

def split_train_test(percent, X, y):###percent=0.75
    train_pct_index = int(percent * len(X))
    X_train, X_test = X[:train_pct_index], X[train_pct_index:]
    y_train, y_test = y[:train_pct_index], y[train_pct_index:] 
    return X_train, X_test, y_train, y_test
    
## classification models
# KNeighborsClassifier
def knn():
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=3)
    return model

def LR():
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    return model

def GaussianNB():####need dense data
    return

def dtc():
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state=42) 
    return model

def rfc():
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42)  
    return model

def abc():
    from sklearn.ensemble import AdaBoostClassifier
    model = AdaBoostClassifier(random_state=42)
    return model

def gbc():
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(random_state=42)
    return model


def lda(): ###need dense data
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    model = LinearDiscriminantAnalysis()
    return model

def qda():
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    model = QuadraticDiscriminantAnalysis()
    return model


def array_value_count(array):
    y = np.bincount(array)
    ii = np.nonzero(y)[0]
    value_count = np.vstack((ii,y[ii])).T
    return value_count
    

