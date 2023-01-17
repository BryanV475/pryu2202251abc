import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def searchNull(data):
    for item in data:
        if item == True: 
            return True
    return False

def getNullColumns(data_for_read):
    for column_name in data_for_read:
        filter = data_for_read[column_name].isnull()
        if(searchNull(filter)):
            yield column_name, filter
            
def getUtilRows(data):
    for  n in range(len(data)):    
        aux = data.iloc[n, :].isnull()
        flag_to_add = True
        for item in aux:
            if item == True:
                flag_to_add = False
                break
        if(flag_to_add):
            yield data.iloc[n, :]

def removeNegatives(x_data, y_data):
    # Crear una lista para almacenar los índices de las filas con valores negativos
    negative_rows = []
    for i, row in y_data.iterrows():
        # Revisar cada valor de la fila actual
        for val in row:
            # Si se encuentra un valor negativo, agregar el índice de la fila a la lista
            if val < 0:
                negative_rows.append(i)
                break
    # Eliminar las filas con valores negativos de ambos dataframes
    x_data = x_data.drop(index=negative_rows)
    y_data = y_data.drop(index=negative_rows)
    return x_data, y_data

            
# obtenemos los datos incompletos
# recorremos las filas y determinamos su utilidad con una bandera, si su valor es False, la fila actual se guarda
def getIncompleteRows(data):
    for  n in range(len(data)):    
        aux = data.iloc[n, :].isnull()
        flag_to_add = True
        for item in aux:
            if item == True:
                flag_to_add = False
                break
        if(flag_to_add is False):
            yield data.iloc[n, :]
            
def saveUtilRows(data_to_walk):
    data = []
    for item in getUtilRows(data_to_walk):
        data.append(item)
    data = pd.DataFrame(data)
    return data

def saveIncompleteRows(data_to_walk):
    data = []
    for item in getIncompleteRows(data_to_walk):
        data.append(item)
    return pd.DataFrame(data)

def normalizeData(data):
    for column in data:
        # Se calcula el valor mínimo y máximo de la columna actual
        min_val = min(data[column])
        max_val = max(data[column])
        
        # Se recorre cada valor de la columna actual y se aplica la fórmula de normalización
        data[column] = [(val - min_val) / (max_val - min_val) for val in data[column]]
    return data

def splitXY(data, y_label_name):
    #print("datos: \n",data)
    x_data = data.drop([y_label_name], axis=1 )
    y_data = pd.DataFrame(data[y_label_name])
    return x_data, y_data


def splitUtilIncompleteData(complete_data, y_label_name):
    c_data = saveUtilRows(complete_data)
    #print(c_data)    
    x_c_data, y_c_data = splitXY(c_data, y_label_name)
    i_data = saveIncompleteRows(complete_data)
    x_i_data, y_i_data = splitXY(i_data, y_label_name)
    x_c_data, y_c_data = removeNegatives(x_c_data, y_c_data)
    y_c_data = normalizeData(y_c_data)
    print(y_c_data)
    return x_c_data, y_c_data, x_i_data, y_i_data

def cleanByError(data_total, data_predicted, data_expected):
    cpy_total = data_total.copy()
    cpy_expected = data_expected.copy()
    c = 0 # contador para recorrer los arreglos (de 0 a n) // no indices
    for i, l in data_total.iterrows():
        error = 100 * (abs(data_expected.iloc[c][0] - data_predicted.iloc[c][0]) / data_expected.iloc[c][0])
      
        if(error>100):  
            cpy_total = cpy_total.drop(i) #  eliminamos los elementos en la posicion i que me generan ruido
            cpy_expected = cpy_expected.drop(i)
        c += 1
    return cpy_total, cpy_expected

data = pd.read_csv('./pagos.csv')

# x_complete_data, y_complete_data, x_incomplete_data, y_incomplete_data
x_c_data, y_c_data, x_i_data, y_i_data = splitUtilIncompleteData(data, "Saldo")

df = saveUtilRows(data)
fig, axes = plt.subplots(nrows=1, ncols=3)

df.plot(ax=axes[0], kind='scatter', y='Saldo', x='Mes', color='red')

lrM = LinearRegression()
lrM.fit(x_c_data, y_c_data)
prediccion = lrM.predict(x_c_data)

x, y = cleanByError(x_c_data, pd.DataFrame(prediccion), y_c_data)

lrMc = LinearRegression()
lrMc.fit(x, y)
prediccionClean = lrM.predict(x_c_data)

plt.subplot(1,3,2)

plt.title('Prediccion 1 - sin limpieza')
plt.xlabel('Mes')
plt.ylabel('Saldo')
plt.plot(x_c_data, prediccion)
plt.plot(x_c_data, y_c_data, 'o')

plt.subplot(1,3,3)

plt.title('Prediccion 2 - con Limpieza')

plt.xlabel('Mes')
plt.ylabel('Saldo')
plt.plot(x, prediccionClean)
plt.plot(x, prediccionClean,'go')

plt.show()
