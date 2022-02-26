# Pacotes
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, completeness_score, fowlkes_mallows_score, homogeneity_score, mutual_info_score, normalized_mutual_info_score, rand_score, v_measure_score, f1_score, precision_score, recall_score
from clusters import *
from data_treatment import data
import pandas as pd 
from spyder_chart import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import random
random.seed(10)

# Dados 
X_train, X_test = data().main()

# Valores previstos 
Forecast = {
    'Metrica':['completeness_score', 'fowlkes_mallows_score', 'mutual_info_score','rand_score'],
    'Agglomerative': [], 'Spectral':[], 'Optics':[], 'Meanshift':[], 'BIRCH':[], 'Dbscan':[], 'Affinity':[],'Kmeans':[]
    }

# Computando as metricas
for method in [Agglomerative,Spectral,Optics,Meanshift,BIRCH,Dbscan,Affinity,Kmeans]:    
    X_pred = method(X_train, X_test)
    for metrics in [completeness_score, fowlkes_mallows_score,normalized_mutual_info_score,adjusted_rand_score]:
        Forecast[method.__name__].append(metrics(X_test.values.T[0],X_pred))


def metrics():
    return Metricas

# Exportando 
metrica = pd.DataFrame(Forecast)
metrica.to_csv("Metricas.csv")

# Criando rede web 
web_plot(metrica)