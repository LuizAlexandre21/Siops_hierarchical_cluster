# Pacotes 
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GridSearchCV
from data_treatment import *
from maps import *
import pickle

## Importandoo os dados 
# Amostras de Treino e Teste  
X_train, X_test = data().main()

# Dataframe 
dados = data().normalized(data().dataframe())

# Estimando o modelo 
params ={
    'eps': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    'metric':['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'],
    'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
}

# Criando o modelo --> cross validation 
model = DBSCAN()

# classificando --> cross validation 
classifier = GridSearchCV(model,params,cv=5,scoring='f1')
classifier.fit(X_train) 

# Melhor estimador 
model = classifier.best_estimator_

# Classificando os clusters por diferentes anos 
# Estatisticas descritivas 
estatisticas = dict()

tabelas =[]

# Filtrando as analises por ano 
for ano in range(2013,2020):

    # Criando o campo no dicionario por ano 
    estatisticas[str(ano)] = dict()
    
    # Filtrando as tabelas por ano 
    df = dados[dados['Ano']==ano]
    ind = df[['Capacidade','Dependência_União','Dependência_Estado','IDH','Dependência_Estado_sus','Dependência_União_sus']]
    ind,rotulate = data.rotulate(1,ind,"IDH")

    # classificando os clusters 
    model.fit_predict(ind)

    # Salando as classificações no banco de dados  
    df['Cluster'] = model.labels_
    tabelas.append(df)
    # Exportando os mapas 
    maps(df,'Codigo',str(ano))

    # Calculando estatisticas descritivas 
    for classification in np.unique(df['Cluster']):

       # Criando a chave por classificação 
        estatisticas[str(ano)][str(classification)] = dict()
        
        # Filtrando por nivel de cluster 
        local = df[df['Cluster']==classification] 

        # Contagem de municipios    
        estatisticas[str(ano)][str(classification)]['Contagem'] = len(local)

        # Municipios 
        estatisticas[str(ano)][str(classification)]['Municipios'] =list(np.unique(local['Municipio']))

        # Media dos indicadores 
        for indicador in ['Capacidade','Dependência_Estado','Dependência_União']:

            # Criando o dicionario para os indicadores 
            estatisticas[str(ano)][str(classification)][str(indicador)]  = dict()

            # Media do indicador
            estatisticas[str(ano)][str(classification)][str(indicador)]['Media'] = np.mean(local[indicador])

            # Desvio Padrão 
            estatisticas[str(ano)][str(classification)][str(indicador)]['Desvio Padrão'] = np.std(local[indicador])

            # quartis 
            estatisticas[str(ano)][str(classification)][str(indicador)]['Quartis'] = [np.quantile(local[indicador], q=0.25),np.quantile(local[indicador], q=0.5),np.quantile(local[indicador], q=0.75)]    

            # Maximo 
            estatisticas[str(ano)][str(classification)][str(indicador)]['Maximo'] = max(local[indicador])

            # Minimo 
            estatisticas[str(ano)][str(classification)][str(indicador)]['Minimo'] = min(local[indicador])

