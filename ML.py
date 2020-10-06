import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv('heartedit.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.20)

classifier = RandomForestClassifier(max_features = 'log2', n_estimators = 250, criterion = 'entropy', random_state = 10)
classifier.fit(X_train, y_train)


pickle.dump(classifier, open('heart.pkl', 'wb'))
model=pickle.load(open('heart.pkl', 'rb'))
print("done")