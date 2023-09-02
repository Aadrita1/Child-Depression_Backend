from fastapi import FastAPI
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = pickle.load(open("./trained_model4.pkl", "rb"))

class InputData(BaseModel):
    school_year: int
    age: int
    bmi: float
    phq_score: int
    depressiveness: bool
    suicidal: bool
    depression_treatment: bool
    gad_score: int
    anxiousness: bool
    anxiety_diagnosis: bool
    anxiety_treatment: bool
    epworth_score: int
    sleepiness: bool
    gender_female: int
    gender_male: int
    who_bmi_Class_I_Obesity: int
    who_bmi_Class_II_Obesity: int
    who_bmi_Class_III_Obesity: int
    who_bmi_Normal: int
    who_bmi_Overweight: int
    who_bmi_Underweight: int
    anxiety_severity_Mild: int
    anxiety_severity_Moderate: int
    anxiety_severity_None_minimal: int
    anxiety_severity_Severe: int
    anxiety_severity_none: int
    depression_severity_Mild: int
    depression_severity_Moderate: int
    depression_severity_Moderately_severe: int
    depression_severity_None_minimal: int
    depression_severity_Severe: int
    depression_severity_none: int


@app.get('/')
def welcome():
    return {
        'success': True,
        'message': 'server of child-depression prediction is running successfully'
    }

@app.post("/predict")
def predict(data: InputData):
  input_features = [data.school_year,
    data.age,
    data.bmi,
    data.phq_score,
    data.depressiveness,
    data.suicidal,
    data.depression_treatment,
    data.gad_score,
    data.anxiousness,
    data.anxiety_diagnosis,
    data.anxiety_treatment,
    data.epworth_score,
    data.sleepiness,
    data.gender_female,
    data.gender_male,
    data.who_bmi_Class_I_Obesity,
    data.who_bmi_Class_II_Obesity,
    data.who_bmi_Class_III_Obesity,
    data.who_bmi_Normal,
    data.who_bmi_Overweight,
    data.who_bmi_Underweight,
    data.anxiety_severity_Mild,
    data.anxiety_severity_Moderate,
    data.anxiety_severity_None_minimal,
    data.anxiety_severity_Severe,
    data.anxiety_severity_none,
    data.depression_severity_Mild,
    data.depression_severity_Moderate,
    data.depression_severity_Moderately_severe,
    data.depression_severity_None_minimal,
    data.depression_severity_Severe,data.depression_severity_none]
  
#   print(input_features)
  input_features_encoded = np.array(input_features).reshape(1,-1)
  prediction = model.predict(input_features_encoded)
# #    return {"prediction": prediction.tolist()}
  predict_depression = prediction.tolist()
  print(predict_depression[0])


  return{
        'status': 'success',
         'prediction': predict_depression[0]
    }