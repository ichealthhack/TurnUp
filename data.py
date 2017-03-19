import numpy as np
import pandas as pds
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, auc


sns.set_style("whitegrid")

data = pds.read_csv('No-show-Issue-Comma-300k-corrected.csv')

data.Gender = data.Gender.apply(lambda x: 1 if x == 'M' else 0)
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
data.DayOfTheWeek = data.DayOfTheWeek.apply(lambda x: days.index(x))

total = 296500
train = int(0.8 * total)
labels_train = data.Status[:train]

feature_list = [
'Age',
'Gender',
'HourOfTheDay',
'DayOfTheWeek',
'Month',
'Status',
'Diabetes',
'Alchoholism',
'Hypertension',
'Handicap',
'Smokes',
'Scholarship',
'Tuberculosis',
'Sms_Reminder',
'AwaitingTime'
]
features_train = data[feature_list].iloc[:train]
print(features_train)
features_test = data[feature_list].iloc[train:]

labels_test = data.Status[train:]
print(labels_test)


# Multinomial naive bayes classification
clf =  MultinomialNB().fit(features_train, labels_train)
print('Accuracy:', round(accuracy_score(labels_test, 
                                        clf.predict(features_test)), 2) * 100, '%')

results = clf.predict_proba(features_test)
print(results)
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(labels_test, [i[1] for i in results])
roc_auc = auc(fpr, tpr)
print(roc_auc)
plt.plot(fpr, tpr, 'ro')
plt.show()



