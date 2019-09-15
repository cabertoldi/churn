import numpy as np
import os
import pandas as pd
import tensorflow as tf


from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def is_churn(customer_values):
    sess = tf.Session()

    with sess.as_default():
        model = _init_model()

        print(customer_values)

        d = _parse_values(customer_values)
        df = pd.DataFrame(data=d)
        
        sc = StandardScaler()
        X = sc.fit_transform(df)

        result = model.predict(X)

        print(result)

        return 'CHURN' if result and result[0] > 0.5 else "NO_CHURN"

    raise Exception("Wasn't possible to classify customer.")

def _parse_values(customer_values):
    customer = {
        'CreditScore': [customer_values['CreditScore']],
        'Geography': [customer_values['Geography']],
        'Gender': [customer_values['Gender']],
        'Age': [customer_values['Age']],
        'Tenure': [customer_values['Tenure']],
        'Balance': [customer_values['Balance']],
        'NumOfProducts': [customer_values['NumOfProducts']],
        'HasCrCard': [customer_values['HasCrCard']],
        'IsActiveMember': [customer_values['IsActiveMember']],
        'EstimatedSalary': [customer_values['EstimatedSalary']]
    }

    return customer

def _init_model():
    if not os.path.isfile('model.weights.best.hdf5'):
        model = _train()
        print('New train!')
    else:
        model = _create_model()
        model.load_weights('model.weights.best.hdf5')
        print('Using network trained!')

    return model

def _create_model():
    model = Sequential()

    model.add(Dense(50, activation = "relu", input_shape=(10,)))
    model.add(Dropout(0.3, noise_shape=None, seed=None))
    model.add(Dense(50, activation = "relu"))
    model.add(Dropout(0.2, noise_shape=None, seed=None))
    model.add(Dense(50, activation = "relu"))

    model.add(Dense(1, activation = "sigmoid"))
    
    model.summary()

    return model

def _train():
    dataset = _prepare_dataset()

    X = dataset.drop(['Exited'],axis=1) 
    y = dataset['Exited']

    sc = StandardScaler()
    X = sc.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    model = _create_model()
    model.compile(optimizer ='adam', loss='binary_crossentropy', metrics = ['accuracy'])

    checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', save_best_only=True)
    model.fit(X_train, y_train, batch_size=64, epochs=100, callbacks=[checkpointer])
    model.save_weights('model.weights.best.hdf5')

    score = model.evaluate(X_test, y_test, verbose=0)
    print('\n', 'Test accuracy:', score[1])

    return model

def _prepare_dataset():
    df = pd.read_csv('./data/Churn_Modelling.csv')

    all_exited = df[df["Exited"] == 1]
    all_current = df[df["Exited"] == 0]
    all_current = all_current.sample(2037)

    frames = [all_exited, all_current]
    dataset = pd.concat(frames)
    print(dataset.shape)

    dataset = dataset.sort_values(by=['RowNumber'])
    dataset = dataset[dataset.columns[3:]]

    geography_labels, geography_uniques = pd.factorize(dataset["Geography"])
    dataset.drop(columns=['Geography'])
    dataset['Geography'] = geography_labels

    gender_labels, gender_uniques = pd.factorize(dataset["Gender"])
    dataset.drop(columns=['Gender'])
    dataset['Gender'] = gender_labels

    return dataset

