import matplotlib.pyplot as plt
import matplotlib
import numpy as np

##all_events bar plot 
fig = plt.figure(figsize=(8,6))
x = [1,2,3,4,5,6]
labels = ["KNeighborsClassifier", "LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier", "AdaBoostClassifier", "GradientBoostingClassifier"]
hight = [0.766, 0.856, 0.719, 0.883, 0.875, 0.890]
yerr = [0.0025, 0.0026, 0.0026, 0.0019, 0.0019, 0.0018]
plt.xticks(x, labels, rotation='vertical')
#colors = ["gold", "lightskyblue", "yellowgreen"]
plt.ylabel("auc_score")
plt.bar(x=x, height=hight, yerr=yerr)
plt.show()
fig.savefig("all_events.png")


##bar  plot male-female
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
labels = ["KNeighborsClassifier", "LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier", "AdaBoostClassifier", "GradientBoostingClassifier"]
men_means = [0.886,	0.856, 0.926, 0.987, 0.873,	0.888]
women_means = [0.879, 0.852, 0.925, 0.989, 0.870, 0.899]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width, label='Men')
rects2 = ax.bar(x + width/2, women_means, width, label='Women')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('roc_auc')
#ax.set_title('Scores by gender')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation='vertical')
ax.legend()
fig.savefig("performance_gender.png")
