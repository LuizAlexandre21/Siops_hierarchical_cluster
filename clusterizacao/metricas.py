# Pacotes
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, completeness_score, fowlkes_mallows_score, homogeneity_score, mutual_info_score, normalized_mutual_info_score, rand_score, v_measure_score, f1_score, precision_score, recall_score
from cluster import *
from data_treatment import data
# Dados 
X_train, X_test, y_train, y_test = data().main()

# Metricas 
Metricas = {
    'Metodo':[],'Rand':[],'Adjusted_rand':[],'Folkes':[],'Adjusted_mutual_info_score':[],'Completeness_score':[],
    'Homogeneity_score':[],'Mutual_info_score':[],'Normalized_mutual_info_score':[], 'V_measure_score':[] }#, 'f1':[],"precision":[],"recall":[]    }

# Computando as metricas
for method in [Agglomerative,Spectral,Optics,Meanshift,BIRCH,Dbscan,Affinity,Knn]:
    Metricas['Metodo'].append(method.__name__)
    y_pred = method(X_train, X_test, y_train, y_test)
    Metricas['Rand'].append(rand_score(y_test.values.T[0], y_pred))
    Metricas['Adjusted_rand'].append(adjusted_rand_score(y_test.values.T[0], y_pred))
    Metricas['Folkes'].append(fowlkes_mallows_score(y_test.values.T[0], y_pred))
    Metricas['Adjusted_mutual_info_score'].append(adjusted_mutual_info_score(y_test.values.T[0], y_pred))
    Metricas['Completeness_score'].append(completeness_score(y_test.values.T[0], y_pred))
    Metricas['Homogeneity_score'].append(homogeneity_score(y_test.values.T[0], y_pred))
    Metricas['Mutual_info_score'].append(mutual_info_score(y_test.values.T[0], y_pred))
    Metricas['Normalized_mutual_info_score'].append(normalized_mutual_info_score(y_test.values.T[0], y_pred))
    Metricas['V_measure_score'].append(v_measure_score(y_test.values.T[0], y_pred))


# Exportando 
pd.DataFrame(Metricas).to_csv("Metricas.csv")