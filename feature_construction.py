from pyspark.sql import *
import helper_function
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from sklearn import metrics
start_time = time.time()

###read csv into DATAFRAME
PATIENTS = helper_function.readCSV("PATIENTS.csv").select("subject_id","gender","expire_flag")
DIAGNOSES_ICD = helper_function.readCSV("DIAGNOSES_ICD.csv").select("subject_id","hadm_id","icd9_code")
DRGCODES = helper_function.readCSV("DRGCODES.csv").select("subject_id","hadm_id","drg_code")
ADMISSIONS = helper_function.readCSV("ADMISSIONS.csv").select("subject_id","admittime")
#SERVICES = helper_function.readCSV("SERVICES.csv").select("subject_id","hadm_id","transfertime")
#TRANSFERS = helper_function.readCSV("TRANSFERS.csv")
#CPTEVENTS = helper_function.readCSV("CPTEVENTS.csv")

#rdd.map(lambda x: [x[i] for i in [0,2,4]), SELECT CERTAIN COLOMUNS
#map to get timestamp
DIAGNOSES_ICD = DIAGNOSES_ICD.join(ADMISSIONS, DIAGNOSES_ICD.hadm_id==ADMISSIONS.hadm_id, how="left")

## get features value for each type
DIAGNOSES_ICD = DIAGNOSES_ICD.groupby(["subject_id", "icd9_code"])
DIAGNOSES_ICD.agg({"*": "count"}).collect()

path = "../data/"
patients = pd.read_csv(path + "PATIENTS.csv")[["SUBJECT_ID","GENDER","EXPIRE_FLAG"]]
diag = pd.read_csv(path + "DIAGNOSES_ICD.csv")[["SUBJECT_ID","ICD9_CODE"]].rename({"ICD9_CODE": "EVENT_ID"},axis=1).dropna()
med = pd.read_csv(path + "DRGCODES.csv")[["SUBJECT_ID","DRG_CODE"]].rename({"DRG_CODE": "EVENT_ID"},axis=1).dropna()
lab = pd.read_csv(path + "LABEVENTS.csv")[["SUBJECT_ID","ITEMID"]].rename({"ITEMID": "EVENT_ID"},axis=1).dropna()

##fearure construction
diag_group = diag.groupby(["SUBJECT_ID","EVENT_ID"]).size().reset_index(name="VALUE")
med_group = med.groupby(["SUBJECT_ID","EVENT_ID"]).size().reset_index(name="VALUE")
lab_group = lab.groupby(["SUBJECT_ID","EVENT_ID"]).size().reset_index(name="VALUE")

###diag id map
diag_id_map = helper_function.id_map(diag)
##run as a whole
index_stop = diag_id_map.ID.max()
med_id_map = helper_function.id_map(med)
med_id_map.ID = med_id_map.ID + index_stop + 1
##run as a whole
index_stop_new = med_id_map.ID.max()
lab_id_map = helper_function.id_map(lab)
lab_id_map.ID = lab_id_map.ID + index_stop_new + 1
###map id to event_id
diag_mapped = helper_function.map_id_event_id(diag_group, diag_id_map)
med_mapped = helper_function.map_id_event_id(med_group, med_id_map)
lab_mapped = helper_function.map_id_event_id(lab_group, lab_id_map)
event_all = pd.concat([diag_mapped,med_mapped,lab_mapped],ignore_index=True)
##get female events
event_all["SEX"] = event_all["SUBJECT_ID"].map(patients.set_index("SUBJECT_ID")["GENDER"])
event_female = event_all.loc[event_all["SEX"]=="F"]
event_male = event_all.loc[event_all["SEX"]=="M"]
## female patient_feature
female_features = helper_function.patient_feature(event_female)
male_feature = helper_function.patient_feature(event_male)
###create patient_flag dic
patient_flag = helper_function.patient_flag_dic(patients)
##save female svmlight
helper_function.svmlight(path, "female.test", female_features, patient_flag)
##save male svmlight
helper_function.svmlight('D:/2020spring/6230/project/FINAL/', "male.test", male_feature, patient_flag)
###create patient_feature dic
patient_features = helper_function.patient_feature(event_all)
###create patient_flag dic
patient_flag = helper_function.patient_flag_dic(patients)
###save svm file
path = 'D:/2020spring/6230/project/FINAL/'
filename = 'all.train'
svm_file = path + filename
helper_function.svmlight(path, filename, patient_features, patient_flag)

print("total time spent is "+ str(time.time()-start_time))



