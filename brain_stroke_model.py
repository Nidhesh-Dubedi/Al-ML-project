import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
import os
import joblib

MODEL_BRAIN_FILE = 'model_brain.pkl'
PIPELINE_BRAIN_FILE = 'pipeline_brain.pkl'

def build_pipeline(num_attribs,cat_attribs):
    num_pipeline = Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ])
    full_pipeline = ColumnTransformer([
        ("num",num_pipeline,num_attribs),
        ("cat",cat_pipeline,cat_attribs)
    ])

    return full_pipeline

if not os.path.exists(MODEL_BRAIN_FILE):
    health = pd.read_csv("brain_stroke.csv")

    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index,test_index in split.split(health,health["hypertension"]):
        health.loc[test_index].to_csv("brain_input.csv",index=False)
        health = health.loc[train_index]

    health_features = health.drop("stroke",axis=1)  
    health_labels = health["stroke"]

    num_attribs = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease']
    cat_attribs = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

    Pipeline = build_pipeline(num_attribs,cat_attribs)
    health_prepared = Pipeline.fit_transform(health_features)

    model = RandomForestClassifier(random_state=42)
    model.fit(health_prepared,health_labels)


    joblib.dump(model,MODEL_BRAIN_FILE)
    joblib.dump(Pipeline,PIPELINE_BRAIN_FILE)

    print("Model Trained and Saved...")

else:
    model = joblib.load(MODEL_BRAIN_FILE)
    Pipeline = joblib.load(PIPELINE_BRAIN_FILE)

    input_data = pd.read_csv("brain_input.csv")   
    transform_input = Pipeline.transform(input_data)
    prediction = model.predict(transform_input)
    input_data["stroke"] = prediction

    input_data.to_csv("brain_output.csv",index=False)
    print("Inference Complete. Result Saved to brain_output.csv") 
