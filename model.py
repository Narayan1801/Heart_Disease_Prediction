import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("heart-disease.csv")

# Importing the RandomForestClassifier Estimator
from sklearn.ensemble import RandomForestClassifier

#Setting up the random seed
np.random.seed(42)

#Preparing the data
X = df.drop("target",axis=1)
y = df["target"]

#Spliting the data

from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Instantiating RandomForestClassifier 
model = RandomForestClassifier( n_estimators= 500,
 min_samples_split = 2,
 min_samples_leaf = 4,
 max_features = 'sqrt' ,
 max_depth = 10
)
model.fit(X_train,y_train)

from sklearn.model_selection import cross_val_score
np.array(cross_val_score(model,X,y,cv=5)).mean()*100

import pickle
pickle.dump(model ,open("heart_model.pkl","wb"))

model = pickle.load(open("heart_model.pkl","rb"))

print(model.predict(([63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1])))