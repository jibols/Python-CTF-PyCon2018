# Complexity : Hard
# K Nearest Neighbors using Scikit-Learn
# Predict the class pof dragon based on color

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

mod1 = pd.read_csv("C:\\Users\\Ajibola Vincent\\Downloads\\ctf\\classified_dragons.csv")
mod2 = pd.read_csv("C:\\Users\\Ajibola Vincent\\Downloads\\ctf\\unclassified_dragons.csv")
#print(mod1.head())
feature_cols = ['Length','Temperature','Weight','Wingspan']
le = preprocessing.LabelEncoder()
mod1['Label'] = le.fit_transform(mod1['Label'])
print(list(le.classes_))
new_result = []
X = mod1[feature_cols]
y = mod1['Label']
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X, y)
result = clf.predict(mod2[feature_cols])
for i in range(result.shape[0]):
    new_result.append(le.classes_[result[i]])
mod2['Label'] = new_result
mod2.to_csv("C:\\Users\\Ajibola Vincent\\Downloads\\ctf\\unclassified_dragons1.csv")
