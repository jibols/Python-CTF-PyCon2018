# Complexity : Hard
# Linear Regression using Scikit-Learn
# Predict the Calc 2 Score for a student who scored 97 in Calc 1
import pandas as pd
from sklearn import linear_model

mod = pd.read_csv("C:\\Users\\Ajibola Vincent\\Downloads\\ctf\\regressive_math_scores.csv")
#print(mod.columns)
X  = mod[['Calc 1 Score']]
y  = mod[['Calc 2 Score']]
model = linear_model.LinearRegression()
model.fit(X, y)
new_pred = model.predict(97)
print('The predicted Calc 2 Score is', new_pred)
print('Coefficients: \n', model.coef_)
