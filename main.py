# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 09:43:39 2021

@author: Yohana Delgado Ramos
"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
from keras.models import Model, load_model
from sklearn.preprocessing import StandardScaler
import pandas as pd

def preprocessing_data(dataset):
    dataset.drop(['Time'], axis=1, inplace=True)
    dataset['Amount'] = StandardScaler().fit_transform(dataset['Amount'].values.reshape(-1,1)) ##normalizacion del dato, valor medio 0 y ds = 1
    dataset = dataset.drop(['Class'], axis=1)
    return dataset

def prediction(dataset):
    from keras.models import load_model
    model1 = load_model('autoencoder_model.h5')


    predit = model1.predict(dataset) #Prediccion
    ecm = np.mean(np.power(dataset-predit,2), axis=1) ##Error
    umbral_fijo = 0.75##sE DEJA EL UMBRAL FIJO, CON EL FIN DE OBTENER LA MAYOR CANTIDAD DE FRAUDES
    Y_pred = [1 if e > umbral_fijo else 0 for e in ecm]
    return Y_pred
    
#app = Flask(__name__)
#port = int(os.getenv("PORT", 8085))
##load the auto encoder instance
# loading whole model

dataset = pd.read_csv("Test.csv")

dataset= preprocessing_data(dataset)

Y_pred=prediction(dataset)
print(Y_pred)
    
    




