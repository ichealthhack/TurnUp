import numpy as np
import pandas as pds
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, auc

def prob_calc(data, variable):
    df = pds.crosstab(index = data[variable], columns = data.Status).reset_index()
    print('the last one')
    print(df)
    print(df[0])
    df['ProbabilityTurnUp'] = df[1] / (df[0] + df[1])
    return df[[variable, 'ProbabilityTurnUp']]

def plot_wrt_variable(data, variable):
    sns.set_style("whitegrid")
    sns.lmplot(data = prob_calc(data, variable), x = variable, y = 'ProbabilityTurnUp', fit_reg = True)
    sns.plt.xlim(0, 100) # Arbitrary limit
    sns.plt.title('Probability of showing up with respect to {}'.format(variable))


# Input data from csv
data = pds.read_csv('No-show-Issue-Comma-300k-corrected.csv')

# Setup DataSegment
data.Gender = data.Gender.apply(lambda x: 1 if x == 'M' else 0)
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
data.DayOfTheWeek = data.DayOfTheWeek.apply(lambda x: days.index(x))


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

# Use 80% for training, 20% for testing
total = 296400
train = int(0.8 * total)

features_train = data[feature_list].iloc[:train]
features_test = data[feature_list].iloc[train:]
labels_test = data.Status[train:]
labels_train = data.Status[:train]

# Multinomial naive bayes classification
clf =  MultinomialNB().fit(features_train, labels_train)
print('Accuracy: ', round(accuracy_score(labels_test, clf.predict(features_test)), 3) * 100, '%')

# AUC
results = clf.predict_proba(features_test)
print(results)
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(labels_test, [i[1] for i in results])
roc_auc = auc(fpr, tpr)
print('AUC: ', roc_auc)
plt.plot(fpr, tpr, 'ro')

# Plot how each variable independently affects the probability of showing up
for variable in feature_list:
    if variable != 'Status':
        plot_wrt_variable(data, variable)

sns.plt.show()
plt.show()



