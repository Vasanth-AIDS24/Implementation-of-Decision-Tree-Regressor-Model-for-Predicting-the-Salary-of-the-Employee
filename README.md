# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.

2.Upload the dataset and check for any null values using .isnull() function.

3.Import LabelEncoder and encode the dataset.

4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5.Predict the values of arrays.

6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7.Predict the values of array.

8.Apply to new unknown values. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Vasanth P
RegisterNumber:212224230295
*/

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree

data = pd.read_csv("Salary_EX7.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data["Position"] = le.fit_transform(data["Position"])

data.head()

x=data[["Position","Level"]]

y=data["Salary"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor,plot_tree

dt=DecisionTreeRegressor()

dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics

mse = metrics.mean_squared_error(y_test,y_pred)

mse

r2=metrics.r2_score(y_test,y_pred)

r2

dt.predict([[5,6]])

plt.figure(figsize=(20, 8))

plot_tree(dt, feature_names=x.columns, filled=True)

plt.show()

```

## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)

<img width="405" height="406" alt="image" src="https://github.com/user-attachments/assets/c7206140-d5d3-4c93-a361-f31a4cf1d0e2" />

<img width="470" height="121" alt="image" src="https://github.com/user-attachments/assets/d9c6380b-4d8b-4e21-bc00-014182b0e55e" />

<img width="355" height="97" alt="image" src="https://github.com/user-attachments/assets/7543fe95-60e5-4577-a747-d3adb657dee4" />

<img width="1504" height="125" alt="image" src="https://github.com/user-attachments/assets/fc7c75cb-865d-4b7c-8c25-f0b5a0dd4f0a" />

<img width="1610" height="547" alt="image" src="https://github.com/user-attachments/assets/62f2b377-1099-4d1d-8637-9507b451f3f6" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
