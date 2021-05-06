# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 20:57:39 2021



Autoencoders : Machine Learning.
Presentado por :  Yohana Delgado Ramos 
                 Milena Beltran B. 
"""
"""
pip install --upgrade pip
pip install --upgrade tensorflow
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno #
import seaborn as sns
from sklearn.metrics import precision_score, accuracy_score, f1_score,  recall_score 
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix, precision_recall_curve
import pickle

"""                     Creacion del modelo           """

""" Para el caso de datos no balanceados, es decir cuando no se tenga informacion suficiente por para predecir 
se recomienda trabajar con la clase que tenga mayor cantidad de datos, en este caso se trabajara con las transacciones 
normales 
# 3.1 La variable "Tiempo" no aporta información. La eliminaremos"""
def autoencoder_model(dataset):
    dataset=preprocessing_data(dataset)
    RANDOM_SEED = 42
    X_train, X_test = train_test_split(dataset, test_size=0.2, random_state=RANDOM_SEED)
    X_train = X_train[X_train.Class == 0] ##Eleccion transacciones normales
    X_train = X_train.drop(['Class'], axis=1) 
    X_train = X_train.values
    
    Y_test = X_test['Class']
    X_test = X_test.drop(['Class'], axis=1)
    X_test = X_test.values
    
    
   # print(X_train.shape)



    np.random.seed(5)
    
    dim_entrada = X_train.shape[1]          # primera capa
    capa_entrada = Input(shape=(dim_entrada,)) ## Entrada 
    
    encoder = Dense(20, activation='tanh')(capa_entrada) ##Primera capa con 20 neuronas
    encoder = Dense(14, activation='relu')(encoder) ##Bottleneck con 14 neuronas
    
    decoder = Dense(20, activation='tanh')(encoder) ##Decoder con 20 neuronas
    decoder = Dense(29, activation='relu')(decoder) ##Salida con 29 neuronas
    
    autoencoder = Model(inputs=capa_entrada, outputs=decoder)
    
    ##Se hace uso del metodo de gradiente descendente 
    sgd = SGD(lr=0.01) ##Tasa de aprendizaje de 0.01
    autoencoder.compile(optimizer='sgd', loss='mse') ## Gradiente , error cuadratico medio
    
    nits =100  ##100 iteraciones
    tam_lote = 32 ##Lote de 32
    autoencoder.fit(X_train, X_train, epochs=nits, batch_size=tam_lote, shuffle=True, validation_data=(X_test,X_test), verbose=1) ##eNTRENAMIENTO DEL MODELO.
    
    ##pickle.dump(autoencoder, open("autoencodermodel.pkl","wb"))
    
    """Evaluacion del modelo"""
    X_pred = autoencoder.predict(X_test) #Prediccion
    ecm = np.mean(np.power(X_test-X_pred,2), axis=1) ##Error
    print(X_pred.shape) ##Tiene el mismo tmaño de los datos de entrada
    
    
    """Validacion a partir de precision y recal definir el umbral para determinar 
    si una transaccion es fraudolenta"""
    
    
    precision, recall, umbral = precision_recall_curve(Y_test, ecm)
    
    plt.plot(umbral, precision[1:], label="Precision",linewidth=5)
    plt.plot(umbral, recall[1:], label="Recall",linewidth=5)
    plt.title('Precicion y recall definicion de umbral')
    plt.xlabel('Umbral')
    plt.ylabel('Precision/Recall')
    plt.legend()
    plt.show()
    
    
    umbral_fijo = 0.75 ##sE DEJA EL UMBRAL FIJO, CON EL FIN DE OBTENER LA MAYOR CANTIDAD DE FRAUDES
    Y_pred = [1 if e > umbral_fijo else 0 for e in ecm] ##Si sobrepasa el umbarl significa que es un fraude. 
    
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    print(conf_matrix)
    
        # create heatmap

    class_names=[False, True] # name  of classes
    fig, ax = plt.subplots(figsize=(7, 6))
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Matriz de confusión')
    plt.ylabel('Real')
    plt.xlabel('Prediccion')
    #Revision de metricasetricas individuales del modelo

    print('Accuracy: {:.2f}'.format(accuracy_score(Y_test, Y_pred)))
    print('Precision: {:.2f}'.format(precision_score(Y_test, Y_pred)))
    print('Recall: {:.2f}'.format(recall_score(Y_test, Y_pred)))
    print('f1_score: {:.2f}'.format(f1_score(Y_test, Y_pred)))
    
    """Guardar el modelo"""
 
        ##serialize weights to HDF5
   # autoencoder.model.save("autoencodermodel.h5")
    autoencoder.save_weights("autoencodermodel.h5")
    print("Saved model to disk")
    # saving whole model
    autoencoder.save('autoencoder_model.h5')
    

def preprocessing_data(dataset):
    dataset.drop(['Time'], axis=1, inplace=True)
    dataset['Amount'] = StandardScaler().fit_transform(dataset['Amount'].values.reshape(-1,1)) ##normalizacion del dato, valor medio 0 y ds = 1
    return dataset
    
    
 

 
"""Importacion del dataset """
dataset = pd.read_csv("creditcard.csv")

##dataset=pd.DataFrame(dataset.data,columns=dataset.feature_names) ##Conversion en dataframe


"""Revision de datos inicial """
print(dataset.head())

dataset.describe()

dataset.info() ##Tipos de columnas del Dataset
##boston_df.plot.box(figsize=(20,10))

"""Numero de registros"""
print(dataset.shape)

""" Verificacion de clases donde 1: transaccion normal y 0 transaccion fraudolenta
Al revisar nos podemos dar cuenta que los datos estan desbalanceados"""

nr_clases = dataset['Class'].value_counts(sort=True)
print(nr_clases)

"""Analisis de los datos """
"""Revision cantidad de datos normales Vs Fraudulentos"""


#Explorando los datos
#Revision de datos faltantes



#Eliminar NAN del dataset. dataset.dropna(inplace=True)

#msno.matrix(dataset) # 

"""Revision de numero de datos faltantes por variable
No se encuentra ningun valor  ausente"""
print(dataset.isna().sum().sort_values())



"""Revision de los datos para la variable Clase"""

plt.figure(figsize=(18,8))
   
y = len(dataset[dataset.Class == 0]),len(dataset[dataset.Class == 1])

cat = ['Normales','Fraudolentos']
plt.bar(cat,y,color = '#7bbdee')
   
plt.xlabel("Clase")   
plt.ylabel("Cantidad")   
plt.title('Valores de frecuencia Transacciones')  


plt.figure(figsize=(12,8))
sns.heatmap(dataset.describe()[1:].transpose(),
            annot=True,linecolor="#0f4b78",
            linewidth=2,cmap=sns.color_palette("muted"))
plt.title("Resumen de atributos")
plt.show()



"""Revision de distribucion de transacciones

"""


# Histogramas 
df_fraud = dataset[dataset.Class == 1] #Transaccion fraudulenta
df_normal = dataset[dataset.Class == 0] #Transaccion Normal

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Montos por tipo de transaccion')

bins = 50

ax1.hist(df_fraud.Amount, bins = bins)
ax1.set_title('Fraude')

ax2.hist(df_normal.Amount, bins = bins, color = '#7bbdee')
ax2.set_title('Normal')

plt.xlabel('Monto ($)')
plt.ylabel('Numero de transacciones')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show();

"""Revision de las transacciones vs tiempo 
No nos ayudan a determinar si es fraude o no """

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Tiempo de transacccion vs monto por tipo de transaccion')

ax1.scatter(df_fraud.Time, df_fraud.Amount, alpha = 0.5,c='#F2545B' , label='Fraude', s=3)


ax2.scatter(df_normal.Time, df_normal.Amount, alpha = 0.5, c='#19323C' , label='Normal', s=3 )


plt.xlabel('Tiempo (s) ')
plt.ylabel('Monto')
plt.legend(loc='upper right')
plt.show()

"""Revision de las demas variables """

 
df_fraud.drop(['Time', 'Amount'], axis=1).hist(bins=30, figsize=(14, 14), color='blue')
plt.suptitle("Histograma para cada atributo Fraude", fontsize=10)
plt.savefig('Histograma_atributo_Fraude.png')
plt.show()



df_normal.drop(['Time', 'Amount'], axis=1).hist(bins=30, figsize=(14, 14), color='blue')
plt.suptitle("Histograma para cada atributo Normales", fontsize=10)
plt.savefig('Histograma_atributo_Normal.png')
plt.show()
autoencoder_model(dataset)


    

