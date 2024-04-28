from flask import Flask, render_template, request, session, url_for, redirect, jsonify,make_response,flash
import pymysql
import random
import smtplib
import string
import math, random
from werkzeug.utils import secure_filename
import os
import pandas as pd

# from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.neighbors import KNeighborsClassifier # K-Nearest Neighbbors
# from sklearn.tree import DecisionTreeClassifier # Decision Tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
import joblib

app = Flask(__name__)

app.config['IMAGE_UPLOADS'] = 'static/hotelimages/'
app.config['IMAGE_UPLOADS_UPDATE'] = 'static/hotelimages1/'

app.secret_key = 'any random string'

def dbConnection():
    connection = pymysql.connect(host="localhost", user="root", password="root", database="criminalprediction",
    charset='utf8')
    return connection

'''def dbConnection():
    try:
        connection = pymysql.connect(host="35.208.147.193", user="inbotics_testing", password="inbotesting", database="inbotics_quizapprohan")
        return connection
    except:
        print("Something went wrong in database Connection")'''


def dbClose():
    try:
        dbConnection().close()
    except:
        print("Something went wrong in Close DB Connection")
        
        
                
con = dbConnection()
cursor = con.cursor()


@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/register')
def register():
    return render_template('register.html') 

@app.route('/login')
def login():
    return render_template('login.html') 

@app.route('/criminal_suspect')
def criminal_suspect():
    return render_template('criminal_suspect.html') 

@app.route('/single')
def single():
    return render_template('single.html') 





@app.route('/userregistration',methods=['POST','GET'])
def userregistration():
    if request.method == "POST":
        details = request.form
        username = details['username']
        email = details['email']
        mobile= details['mobno']
        password1 = details['password']
        address = details['address']
  
        sql2  = "INSERT INTO register(username,email,mobileno,address,password) VALUES (%s, %s, %s, %s, %s)"
        val2 = (str(username),  str(email),str(mobile), str(address), str(password1))
        cursor.execute(sql2,val2) 
        con.commit()
        print("username",username)
       
        
        return render_template('login.html') 
        
    

@app.route('/userlogin', methods=["GET","POST"])
def userlogin():
    msg = ''
    if request.method == "POST": 
        
          
            username = request.form.get("username")
            print ("username",username)
            password = request.form.get("password")
            
            con = dbConnection()
            cursor = con.cursor()
            cursor.execute('SELECT * FROM register WHERE username = %s AND password = %s ' , (username, password))
            result = cursor.fetchone()
            print ("result",result)
            if result:
                session['user'] = result[0]
                return render_template('homepage.html') 
            else:
                msg = 'Incorrect username/password!'
                return msg
   
    return render_template('index.html')


model_filename1 = "models/KNeighbors_Classifier_Classify_model.joblib"
print(model_filename1)
loaded_model1 = joblib.load(model_filename1)

@app.route('/criminalpredict',methods=['POST','GET'])
def criminalpredict():
    if request.method == "POST":
        details = request.form
      
        NRCH17_2 = details['NRCH17_2']
        IRHH65_2 = details['IRHH65_2']
        GRPHLTIN= details['GRPHLTIN']
        HLTINNOS = details['HLTINNOS']
        HLCLAST = details['HLCLAST']
        IRMEDICR = details['IRMEDICR']
        IRPRVHLT = details['IRPRVHLT']
        IROTHHLT = details['IROTHHLT']
        IRFAMSOC = details['IRFAMSOC']
        IRPINC3 = details['IRPINC3']
        POVERTY3 = details['POVERTY3']
      
        
        selected_col = ['NRCH17_2','IRHH65_2','GRPHLTIN','HLTINNOS','HLCLAST','IRMEDICR',
               'IRPRVHLT','IROTHHLT','IRFAMSOC','IRPINC3','POVERTY3']
        encoded_df=pd.read_csv("preprocess.csv")[selected_col]
        to_scale = [col for col in encoded_df.columns if encoded_df[col].max()>1]
        scaler = RobustScaler()
        scaled =scaler.fit_transform(encoded_df[to_scale])
        scaled = pd.DataFrame(scaled, columns=to_scale)
                
        data_list = [
          float(NRCH17_2),
          float(IRHH65_2),
          float(GRPHLTIN),
          float(HLTINNOS),
          float(HLCLAST),
          float(IRMEDICR),
          float(IRPRVHLT),
          float(IROTHHLT),
          float(IRFAMSOC),
          float(IRPINC3),
          float(POVERTY3)]
    
        # Creating DataFrame from the list
        df = pd.DataFrame([data_list], columns=[
            'NRCH17_2', 'IRHH65_2', 'GRPHLTIN', 'HLTINNOS', 'HLCLAST',
            'IRMEDICR', 'IRPRVHLT', 'IROTHHLT', 'IRFAMSOC', 'IRPINC3', 'POVERTY3'
        ])
        
        df2=scaler.transform(df[to_scale])    
        df2=pd.DataFrame(df2,columns=to_scale)
        new_predictions = loaded_model1.predict(df2)
        
        print("=============================")
        print(new_predictions)
        print("=============================")
        
        if new_predictions[0] == 0:
            print("Non criminal")
            prediction="Non criminal"
            flash(prediction, 'success')
            return render_template('single.html',prediction=prediction) 
        else:
            print("Criminal")
            prediction="Criminal"
            flash(prediction, 'danger')
            return render_template('single.html',prediction=prediction) 
        
     
    return render_template('single.html') 
   
        
    


if __name__ == "__main__":
    app.run("0.0.0.0")