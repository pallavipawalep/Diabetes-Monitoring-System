from django.http import HttpResponse
from django.shortcuts import render
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
import joblib
import pandas as pd
import numpy as np

# Create your views here.
def home(request):
    return render(request, "home.html")

def prediction(request):
    fastinSugar = request.POST.get("fastinSugar")
    postprandial = request.POST.get("postprandial")
    hba1c = request.POST.get("hba1c")
    hba1c = float(hba1c)
    print(fastinSugar)
    print(postprandial)
    print(hba1c)
    dataset = pd.read_csv(r'C:/Users/palla/Dropbox/My PC (LAPTOP-MEMQ4MVO)/Documents/diabetes_patient_monitor/service/Diabetes_Monitoring_data-1.csv')


    #Splitting data into X and y
    X = dataset.drop('Output',axis=1)
    X = X.drop('Patient Record', axis=1)

    Y = dataset['Output']

    #Splitting dataset into training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

    #Building classifier
    classifier = RandomForestClassifier()
    classifier.fit(X_train.values, Y_train)

    #Prediction
    prediction = classifier.predict(X_test)
    print("The testing dataset output:")
    print(prediction)

    print("The accuracy of the prediction:")
    print(classifier.score(X_test, Y_test))

    ans = classifier.predict([[fastinSugar,postprandial,hba1c]])
    A = "Your diabetes is well controlled. Maintain the same consistency. Exercise 20 minutes per day for 5 times in a week. Maintain your diet and Cheer up!!"
    B = "You need to take precautions to control diabetes. Do not miss your antidiabetic drugs. Exercise 20 minutes per day for 5 times in a week. Maintain normal levels of your blood pressure and Cholesterol."
    C = "!! YOU must take precautions and follow rules to control diabetes. Consult to your attending physician regularly. Do not miss your antidiabetic drugs. Exercise 20 minutes per day for 5 times in a week. Maintain normal levels of your blood pressure and Cholesterol. Do foot examination daily for any injury to foot or decresed sensation of foot to avoid diabetic foot. Do not go barefoot to avoid injury to foot- Footcare important. Eyecare -checkup if blurring of vision or yearly to avoid diabetic retinopathy. "
    if(ans == 'A'):
        return render(request,"home.html",{ 'error' : True,'message': A})
    elif(ans == 'B'):
        return render(request,"home.html",{ 'error' : True,'message': B})
    else:
        return render(request,"home.html",{ 'error' : True,'message': C})






    