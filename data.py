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


