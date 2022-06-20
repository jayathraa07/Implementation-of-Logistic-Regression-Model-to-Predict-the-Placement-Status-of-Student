# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: JAYATHRAA V
RegisterNumber: 212219220018

import pandas as pd
df=pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Semster 2/Intro to ML/Placement_Data.csv")
df.head()
df.tail()
df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
df1.head()
df1.isnull().sum()
#to check any empty values are there
df1.duplicated().sum()
#to check if there are any repeted values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df1["gender"] = le.fit_transform(df1["gender"])
df1["ssc_b"] = le.fit_transform(df1["ssc_b"])
df1["hsc_b"] = le.fit_transform(df1["hsc_b"])
df1["hsc_s"] = le.fit_transform(df1["hsc_s"])
df1["degree_t"] = le.fit_transform(df1["degree_t"])
df1["workex"] = le.fit_transform(df1["workex"])
df1["specialisation"] = le.fit_transform(df1["specialisation"])
df1["status"] = le.fit_transform(df1["status"])
df1
x=df1.iloc[:,:-1]
x
y = df1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.09,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
#liblinear is library for large linear classification
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
print(lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]))

```

## Output:
Original data(first five columns):
![image](https://user-images.githubusercontent.com/107881970/174659797-20df9f54-3776-42e9-b96e-fdfd74b1417d.png)

Data after dropping unwanted columns(first five):
![image](https://user-images.githubusercontent.com/107881970/174659913-c3bf1b25-9632-459a-9338-85445f153306.png)

Checking the presence of null values:
![image](https://user-images.githubusercontent.com/107881970/174659938-eb89d945-6e11-4736-b83b-7d32364e24ca.png)

Checking the presence of duplicated values:
![image](https://user-images.githubusercontent.com/107881970/174660002-89cdd175-50eb-49c4-898f-4ff0ba626cee.png)

Data after Encoding:
![image](https://user-images.githubusercontent.com/107881970/174660044-150d4e50-f315-4364-9462-69946d400a97.png)

X Data:
![image](https://user-images.githubusercontent.com/107881970/174660075-1ae7d923-274b-4f6b-b78a-7e31b7a5d1c3.png)

Y Data:
![image](https://user-images.githubusercontent.com/107881970/174660113-07d9b49b-d55f-4c20-a839-b753069a7644.png)

Predicted Values:
![image](https://user-images.githubusercontent.com/107881970/174660135-6f24a4c0-0cc9-4e45-b13b-b05fa54f3685.png)

Accuracy Score:
![image](https://user-images.githubusercontent.com/107881970/174660162-2927236c-494d-45bb-8337-f36cbe6928cf.png)

Confusion Matrix:
![image](https://user-images.githubusercontent.com/107881970/174660180-b5a2004e-947a-45c0-a40e-5c7bcfd68ce9.png)

Classification Report:
![image](https://user-images.githubusercontent.com/107881970/174660203-23b24345-18b8-4bea-8c6b-6018523e7e44.png)

Predicting output from Regression Model:
![image](https://user-images.githubusercontent.com/107881970/174660231-9c0a674c-1b1c-4626-89c0-ad9809eaf4fd.png)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
