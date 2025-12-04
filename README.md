# Diebities-prediction
I created a binary classification model that predicts whether a person is diabetic (1) or not diabetic (0).



import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

Data  Clollection and Analysis

diabetes_dataset=pd.read_csv("/content/diabetes.csv")

diabetes_dataset.head()

diabetes_dataset.shape

diabetes_dataset.describe()

diabetes_dataset["Outcome"].value_counts()

0>-Non Diabetic
1>-Diabitic

diabetes_dataset.groupby("Outcome").mean()

x=diabetes_dataset.iloc[:,:-1]
y=diabetes_dataset["Outcome"]

Data Standardization

scaler=StandardScaler()
scaler.fit(x)

Standardizad_data=scaler.transform(x)

x=Standardizad_data
y=diabetes_dataset["Outcome"]


print(x)
print(y)

Train Test Split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)

Training The Model

classifier=SVC(kernel="linear")
classifier.fit(x_train,y_train)

Model Evaluation

Accuracy Score

x_train_prediction=classifier.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)

print("Accuracy Score of the Training Data:",training_data_accuracy)

Testing Accuracy

x_test_prediction=classifier.predict(x_test)
testing_data_accuracy=accuracy_score(x_test_prediction,y_test)

print("Accuracy Score of the Testing Data:",testing_data_accuracy)

Making  a predictive system

input_data=(4,110,92,0,0,37.6,0.191,30)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
std_data=scaler.transform(input_data_reshaped)
print(std_data)

prediction=classifier.predict(std_data)
print(prediction)

if(prediction[0]==0):
  print("The person is not Diabetic")
else:
  print("The person is Diabetic")

