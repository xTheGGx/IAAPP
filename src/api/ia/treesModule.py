from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import export_text
from multiprocessing import current_process
import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
import os
# Para generar y almacenar los gráficos dentro del cuaderno
import yfinance as yf

current_dir = os.path.dirname(os.path.abspath(__file__))


def createGrafProg():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chart_path = os.path.normpath(os.path.join(current_dir, '../../static/img/charts/treeProg.png'))
    DataAmazon = yf.Ticker('AMZN')
    AmazonHist = DataAmazon.history(start = '2019-1-1', end = '2023-5-25', interval='1d')
    plt.figure(figsize=(20, 5))
    plt.plot(AmazonHist['Open'], color='purple', marker='+', label='Open')
    plt.plot(AmazonHist['High'], color='blue', marker='+', label='High')
    plt.plot(AmazonHist['Low'], color='orange', marker='+', label='Low')
    plt.plot(AmazonHist['Close'], color='green', marker='+', label='Close')
    plt.xlabel('Fecha')
    plt.ylabel('Precio de las acciones')
    plt.title('Amazon')
    plt.grid(True)
    plt.legend()
    plt.savefig(chart_path, bbox_inches='tight')

def execProgTree(depth,split,leaf,open,high,low):
    DataAmazon = yf.Ticker('AMZN')
    AmazonHist = DataAmazon.history(start = '2019-1-1', end = '2023-5-25', interval='1d')
    MatrizDatos = AmazonHist.drop(columns = ['Volume', 'Dividends', 'Stock Splits'])
    MatrizDatos = MatrizDatos.dropna()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chart_path = os.path.normpath(os.path.join(current_dir, '../../static/img/charts/treeProg.png'))
    X = np.array(MatrizDatos[['Open',
                        'High',
                        'Low']])
    Y = np.array(MatrizDatos[['Close']])
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                        test_size = 0.2, 
                                                                        random_state = 0, 
                                                                        shuffle = True)      
    ProgAD = DecisionTreeRegressor(max_depth=depth, min_samples_split=split, min_samples_leaf=leaf, random_state=0)
    ProgAD.fit(X_train, Y_train)

    Y_Pronostico = ProgAD.predict(X_test)

    Valores = pd.DataFrame(Y_test, Y_Pronostico)

    score=r2_score(Y_test, Y_Pronostico)

    plt.figure(figsize=(20, 5))
    plt.plot(Y_test, color='red', marker='+', label='Real')
    plt.plot(Y_Pronostico, color='green', marker='+', label='Estimado')
    plt.xlabel('Fecha')
    plt.ylabel('Precio de las acciones')
    plt.title('Pronóstico de las acciones de Amazon')
    plt.grid(True)
    plt.legend()
    plt.savefig(chart_path, bbox_inches='tight')

    Reporte = export_text(ProgAD, feature_names = ['Open', 'High', 'Low'])
    PrecioAccion=pd.DataFrame({'Open': [open],'High':[high],'Low':[low]})
    Predic = str(ProgAD.predict(PrecioAccion))

    return Reporte, score, Predic
