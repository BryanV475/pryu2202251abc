import pandas as pd
import matplotlib.pyplot as plt
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

def splitXY(data, y_label_name):
    #print("datos: \n",data)
    x_data = data.drop([y_label_name], axis=1 )
    y_data = pd.DataFrame(data[y_label_name])
    return x_data, y_data

def splitUtilIncompleteData(complete_data, y_label_name):
    c_data = saveUtilRows(complete_data)
    print(c_data)    
    x_c_data, y_c_data = splitXY(c_data, y_label_name)
    i_data = saveIncompleteRows(complete_data)
    x_i_data, y_i_data = splitXY(i_data, y_label_name)
    return x_c_data, y_c_data, x_i_data, y_i_data

data = pd.read_csv('./pagos.csv')

# x_complete_data, y_complete_data, x_incomplete_data, y_incomplete_data
x_c_data, y_c_data, x_i_data, y_i_data = splitUtilIncompleteData(data, "Saldo")

df = saveUtilRows(data)
fig, axes = plt.subplots(nrows=1, ncols=3)

df.plot(ax=axes[0], kind='scatter', y='Saldo', x='Mes', color='red')

lrM = LinearRegression()
lrM.fit(x_c_data, y_c_data)
prediccion = lrM.predict(x_c_data)

plt.subplot(1,3,2)
plt.title('Prediccion 1 - sin limpieza')
plt.xlabel('Mes');
plt.ylabel('Saldo');
plt.plot(x_c_data, prediccion)
plt.plot(x_c_data, y_c_data, 'o')

plt.subplot(1,3,3)
plt.title('Prediccion 2 - con Limpieza')
plt.plot(x_c_data, prediccion,'go')
plt.xlabel('Mes');
plt.ylabel('Saldo');
plt.plot(x_c_data, prediccion)
#plt.xlabel('Mes')
#plt.ylabel('Saldo')
plt.show()
