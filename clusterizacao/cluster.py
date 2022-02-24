from sklearn.cluster import AgglomerativeClustering, SpectralClustering, OPTICS, MeanShift, Birch, DBSCAN, AffinityPropagation,KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, completeness_score, fowlkes_mallows_score, homogeneity_score, mutual_info_score, normalized_mutual_info_score, rand_score, v_measure_score


# Agglomerative Clustering 
# TODO:Grid Search 
def Agglomerative(X_train, X_test, y_train, y_test):
    classifier = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
    classifier.fit(X_train,y_train) 

    # Prevendo os modelos 
    y_pred = classifier.fit_predict(X_test)

    return y_pred 

# Spectral Clustering
def Spectral(X_train, X_test, y_train, y_test):
    # GridSearch
    # Parametros
    params = {
        'n_clusters':[2,3,4,5,6,7,8],
        'eigen_solver':['arpack', 'lobpcg', 'amg'],
        'affinity':['nearest_neighbors','rbf']
    }

    # Scoring 
    model =  SpectralClustering()
    classifier = GridSearchCV(model,params,cv=5,scoring='f1')
    classifier.fit(X_train,y_train) 
    
    # Melhor estimador 
    model = classifier.best_estimator_

    # ajustando o modelo
    model.fit(X_train,y_train)
    
    # Prevendo  
    y_pred = model.fit_predict(X_test)

    return y_pred 

# Optics Clustering
def Optics(X_train, X_test, y_train, y_test):
    # GridSearch
    # Parametros
    params ={
        'max_eps': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
        'metric':['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'],
        'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
    }

    model =  OPTICS()
    classifier = GridSearchCV(model,params,cv=5,scoring='f1')
    classifier.fit(X_train,y_train) 
    
    # Melhor estimador 
    model = classifier.best_estimator_

    # ajustando o modelo
    model.fit(X_train,y_train)
    
    # Prevendo  
    y_pred=model.fit_predict(X_test)

    return y_pred 

# MeanShift Clustering 
def Meanshift(X_train, X_test, y_train, y_test):
    # GridSearch
    # Parametros
    params ={
        'bandwidth': [2,3,4,5,6,7,8,9],
    }

    model =  MeanShift()
    classifier = GridSearchCV(model,params,cv=5,scoring='f1')
    classifier.fit(X_train,y_train) 
    
    # Melhor estimador 
    model = classifier.best_estimator_

    # ajustando o modelo
    model.fit(X_train,y_train)
    
    # Prevendo  
    y_pred=model.fit_predict(X_test)

    return y_pred 

#Birch
def BIRCH(X_train, X_test, y_train, y_test):
    # GridSearch 
    # Parametros 
    params ={
        'threshold': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
        'n_clusters': [2,3,4,5,6,7,8,9]
        }
    
    model =  Birch()
    classifier = GridSearchCV(model,params,cv=5,scoring='f1')
    classifier.fit(X_train,y_train) 
    
    # Melhor estimador 
    model = classifier.best_estimator_

    # ajustando o modelo
    model.fit(X_train,y_train)
    
    # Prevendo  
    y_pred=model.fit_predict(X_test)

    return y_pred 

# Dbscan
def Dbscan(X_train, X_test, y_train, y_test):
    # GridSearch 
    # Parametros 
    params ={
        'eps': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
        'metric':['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'],
        'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
    }
    model = DBSCAN()
    classifier = GridSearchCV(model,params,cv=5,scoring='f1')
    classifier.fit(X_train,y_train) 
    
    # Melhor estimador 
    model = classifier.best_estimator_

    # ajustando o modelo
    model.fit(X_train,y_train)
    
    # Prevendo  
    y_pred=model.fit_predict(X_test)

    return y_pred   

# AffinityPropagation
def Affinity(X_train, X_test, y_train, y_test):
    # GridSearch 
    # Parametros 
    params = {
        'damping': [0.5,0.6,0.7,0.8,0.9,1],
    }
    model = AffinityPropagation()
    classifier = GridSearchCV(model,params,cv=5,scoring='f1')
    classifier.fit(X_train,y_train) 

    # Melhor estimador 
    model = classifier.best_estimator_

    # ajustando o modelo
    model.fit(X_train,y_train)
    
    # Prevendo  
    y_pred=model.fit_predict(X_test)

    return y_pred  

# KNeighborsClassifier
def Knn(X_train, X_test, y_train, y_test):
    # GridSearch 
    # Parametros 
    params = {
        'n_neighbors':[1,2,3,4,5,6,7,8],
        'weights':['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        
    }
    model = KNeighborsClassifier()
    classifier = GridSearchCV(model,params,cv=5,scoring='f1')
    classifier.fit(X_train,y_train) 
    
    # Melhor estimador 
    model = classifier.best_estimator_

    # ajustando o modelo
    model.fit(X_train,y_train)
    
    # Prevendo  
    y_pred=model.predict(X_test)

    return y_pred  

# Kmeans 
def Kmeans(X_train,X_test,y_train,y_test):
    # GridSearch 
    # Parametros 
    params = {
        'n_clusters':[2,3,4,5,6,7,8],
    }
    model = KMeans()
    classifier = GridSearchCV(model,params,cv=5,scoring='f1')
    classifier.fit(X_train,y_train) 
    
    # Melhor estimador 
    model = classifier.best_estimator_

    # ajustando o modelo
    model.fit(X_train,y_train)
    
    # Prevendo  
    y_pred=model.predict(X_test)

    return y_pred  
