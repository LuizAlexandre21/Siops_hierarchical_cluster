# Pacotes
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, completeness_score, fowlkes_mallows_score, homogeneity_score, mutual_info_score, normalized_mutual_info_score, rand_score, v_measure_score, f1_score, precision_score, recall_score
from cluster import *
from data_treatment import data
import pandas as pd 
from spyder_chart import *

# Dados 
X_train, X_test, y_train, y_test = data().main()

# Valores previstos 
Forecast = {
    'Metrica':['adjusted_mutual_info_score', 'adjusted_rand_score', 'completeness_score', 'fowlkes_mallows_score', 'homogeneity_score', 'mutual_info_score','normalized_mutual_info_score', 'rand_score', 'v_measure_score'],
    'Agglomerative': [], 'Spectral':[], 'Optics':[], 'Meanshift':[], 'BIRCH':[], 'Dbscan':[], 'Affinity':[],'Knn':[]
    }

# Computando as metricas
for method in [Agglomerative,Spectral,Optics,Meanshift,BIRCH,Dbscan,Affinity,Knn]:    
    y_pred = method(X_train, X_test, y_train, y_test)
    for metrics in [adjusted_mutual_info_score, adjusted_rand_score, completeness_score, fowlkes_mallows_score, homogeneity_score, mutual_info_score, normalized_mutual_info_score, rand_score, v_measure_score]:
        Forecast[method.__name__].append(metrics(y_test.values.T[0],y_pred))


def metrics():
    return Metricas

# Exportando 
metrica = pd.DataFrame(Forecast)
metrica.to_csv("Metricas.csv")

# Criando rede web 
web_plot(metrica)