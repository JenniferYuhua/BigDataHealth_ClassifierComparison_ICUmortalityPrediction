import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
path = "../data/"
patients = pd.read_csv(path + "PATIENTS.csv")[["SUBJECT_ID","GENDER","EXPIRE_FLAG"]]
sex = patients.GENDER.value_counts()
###count number of F and M in gebder
death_alive = patients.EXPIRE_FLAG.value_counts()

diag = pd.read_csv(path + "DIAGNOSES_ICD.csv")[["SUBJECT_ID","ICD9_CODE"]].rename({"ICD9_CODE": "EVENT_ID"},axis=1).dropna()
med = pd.read_csv(path + "DRGCODES.csv")[["SUBJECT_ID","DRG_CODE"]].rename({"DRG_CODE": "EVENT_ID"},axis=1).dropna()
lab = pd.read_csv(path + "LABEVENTS.csv")[["SUBJECT_ID","ITEMID"]].rename({"ITEMID": "EVENT_ID"},axis=1).dropna()

import matplotlib.pyplot as plt

##events pie plot 
fig = plt.figure(figsize=(8,6))
labels = "Lab", "Med", "Diag"
sizes = [27854055, 125557, 651000]
colors = ["gold", "lightskyblue", "yellowgreen"]
plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()
fig.savefig("events.png")


##sex pie plot 
fig = plt.figure(figsize=(8,6))
labels = "Male", "Female"
sizes = [26121, 20399]
colors = ["gold", "lightskyblue"]
plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()
fig.savefig("sex.png")


##death_alive pie plot 
fig = plt.figure(figsize=(8,6))
labels = "Alive", "Deceased"
sizes = [30761, 15759]
colors = ["gold", "lightskyblue"]
plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()
fig.savefig("death_alive.png")
