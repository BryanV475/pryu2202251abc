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

data = pd.read_csv('./estaturas.csv')

plt.plot(data['Edad'],data['Estatura'], 'ro')
plt.show()
