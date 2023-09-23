# Import writer class from csv module
from csv import writer
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    dataFrame = pd.read_csv('cancer-patient-data-sets.csv')
    Classifier = LungCancerRiskPrediction(dataFrame)
    # Extract form data for all attributes
    age = int(request.form['age'])
    gender = request.form['gender']
    air_pollution = int(request.form['air-pollution'])
    alcohol_use = int(request.form['alcohol-use'])
    dust_allergy = int(request.form['dust-allergy'])
    occu_hazards = int(request.form['occu-hazards'])
    genetic_risk = int(request.form['genetic-risk'])
    chronic_lung_disease = int(request.form['lung-disease'])
    balanced_diet = int(request.form['balanced-diet'])
    obesity = int(request.form['obesity'])
    smoking = int(request.form['smoking'])
    passive_smoker = int(request.form['passive-smoker'])
    chest_pain = int(request.form['chest-pain'])
    coughing_of_blood = int(request.form['coughing-blood'])
    fatigue = int(request.form['fatigue'])
    weight_loss = int(request.form['weight-loss'])
    shortness_of_breath = int(request.form['shortness-of-breath'])
    wheezing = int(request.form['wheezing'])
    swallowing_difficulty = int(request.form['swallowing-difficulty'])
    clubbing_of_finger_nails = int(request.form['clubbing-finger-nails'])
    frequent_cold = int(request.form['frequent-cold'])
    dry_cough = int(request.form['dry-cough'])
    snoring = int(request.form['snoring'])
  


    # Create a numpy array with the form data
    personDetails = np.array([age, coughing_of_blood, dust_allergy, passive_smoker, occu_hazards, air_pollution, chronic_lung_disease, shortness_of_breath, dry_cough, snoring, swallowing_difficulty]).reshape(1, -1)

    # Predict the risk level
    predicted_level = predictForOnePerson(Classifier, personDetails)

    # level_colors = {1: 'low', 2: 'medium', 3: 'high'}
    # predicted_level_css = level_colors[predicted_level]
    genderId = 1 if gender=='Male' else 2
    currRow = dataFrame.shape[0]+1
    PatientId= 'P'+str(currRow)

    newRow = [
        currRow,PatientId,age, genderId, air_pollution, alcohol_use, dust_allergy, occu_hazards,
        genetic_risk, chronic_lung_disease, balanced_diet, obesity, smoking,
        passive_smoker, chest_pain, coughing_of_blood, fatigue, weight_loss,
        shortness_of_breath, wheezing, swallowing_difficulty, clubbing_of_finger_nails,
        frequent_cold, dry_cough, snoring,predicted_level]

    print(newRow)
    with open("./cancer-patient-data-sets.csv", 'a') as f_object:
    
        # Pass this file object to csv.writer()
        # and get a writer object
        writer_object = writer(f_object)
    
        # Pass the list as an argument into
        # the writerow()
        writer_object.writerow(newRow)
    
        # Close the file object
        f_object.close()

    return render_template('predict.html', predicted_level=predicted_level, result_color=predicted_level)




def LungCancerRiskPrediction(df):
   
    df=df.replace({'Level':{'Low': 1, 'Medium': 2, 'High': 3}})
    
    df=df[['Age','Coughing of Blood','Dust Allergy','Passive Smoker','OccuPational Hazards','Air Pollution','chronic Lung Disease','Shortness of Breath','Dry Cough','Snoring','Swallowing Difficulty','Level']]
    X=df.drop('Level',axis=1)

    y=df['Level']

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=1)

    Classifier = LogisticRegression(solver='liblinear')
    Classifier.fit(X_train,y_train)

    
    return Classifier 


def predictForOnePerson(Classifier ,personDetails):#return string as output

    level = {1:'Low', 2:'Medium', 3:'High'}

    predictedLevel = Classifier.predict(personDetails)
 
    return level[list(predictedLevel)[0]]
 


if __name__ == '__main__':
    app.run(debug=True)
