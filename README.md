# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the required libraries for logistic regression.
2. Load the placement dataset and remove unnecessary columns and duplicate values.
3. Convert categorical data into numerical values using Label Encoding.
4. Split the dataset into training and testing sets, then train the Logistic Regression model.
5. Predict the results, calculate accuracy, and display the confusion matrix and classification report.

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

**Developed by: SRIDHAR C<br>
RegisterNumber: 212225040425**

## Program:
```py


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


#load the data from the given question
df=pd.read_csv("Placement_Data.csv")
df
df1=df.copy()
df1
#drop the unnecessary datas
df1=df1.drop(['sl_no','salary'],axis=1)
df1.isnull().sum()
df1.duplicated().sum()
df1

#convert the categorical variables
le=LabelEncoder()
df1['gender']=le.fit_transform(df1['gender'])
df1['ssc_b']=le.fit_transform(df1['ssc_b'])
df1['hsc_b']=le.fit_transform(df1['hsc_b'])
df1['hsc_s']=le.fit_transform(df1['hsc_s'])
df1['degree_t']=le.fit_transform(df1['degree_t'])
df1['workex']=le.fit_transform(df1['workex'])
df1['specialisation']=le.fit_transform(df1['specialisation'])
df1['status']=le.fit_transform(df1['status'])
df1
x=df1.iloc[:,:-1]
x
y=df1['status']
y

#train and test the model
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)


#print the resuts:
print("Predicted Values: ",y_pred)
print("Accuracy score: ",accuracy)
print("Confusion matrix:\n",confusion)
print("\nClassification Report:\n",cr)

```

## Output:

<img width="942" height="383" alt="Screenshot 2026-05-14 111927" src="https://github.com/user-attachments/assets/dba1c145-7515-4633-a6e0-2a7a08168cbe" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming is executed successfully.
