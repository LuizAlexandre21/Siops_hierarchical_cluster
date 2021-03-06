from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from data_treatment import data 
import matplotlib.pyplot as plt 
import pandas as pd 

# Importando os dados 
dados = data().dataframe()
# Filtrando por anos 
dic = {
    'Capacidade':{
        'Grupo_1':{'Ano':[],'Valor':[]},'Grupo_2':{'Ano':[],'Valor':[]},'Grupo_3':{'Ano':[],'Valor':[]}
        },
    'Dependência_União':{
        'Grupo_1':{'Ano':[],'Valor':[]},'Grupo_2':{'Ano':[],'Valor':[]},'Grupo_3':{'Ano':[],'Valor':[]}
        },
    'Dependência_Estado':{
        'Grupo_1':{'Ano':[],'Valor':[]},'Grupo_2':{'Ano':[],'Valor':[]},'Grupo_3':{'Ano':[],'Valor':[]}
        }
    }
for ano in range(2013,2019):
    
    # Filtrando os dados 
    data = dados[dados['Ano']==ano]
    

    X = data[['Capacidade','Dependência_União','Dependência_Estado']] 
    Y_pre = data['IDH']
    Y_pre=Y_pre.replace('Medio',1)
    Y_pre=Y_pre.replace('Alto',0)
    Y_pre=Y_pre.replace('Baixo',2)
    Y = Y_pre

    
    # Criando amostra de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
    
    # Criando o classificador 
    # GridSearch 
    # Parametros 
    classifier = KMeans(n_clusters=3)
    classifier.fit(X_train,y_train) 
    # Prevendo  
    y_pred=classifier.predict(X_test)

    # Capturando o centroid
    centroid = classifier.cluster_centers_

    # Salvando
    for line in range(0,3):
        for col in range(0,3):
            if line == 0 and col == 0:
                dic['Capacidade']['Grupo_1']['Valor'].append(centroid[line][col])
                dic['Capacidade']['Grupo_1']['Ano'].append(ano)
            elif line == 0 and col == 1:
                dic['Dependência_União']['Grupo_1']['Valor'].append(centroid[line][col])
                dic['Dependência_União']['Grupo_1']['Ano'].append(ano)
            elif line == 0 and col == 2:
                dic['Dependência_Estado']['Grupo_1']['Valor'].append(centroid[line][col])
                dic['Dependência_Estado']['Grupo_1']['Ano'].append(ano)
            elif line == 1 and col == 0:
                dic['Capacidade']['Grupo_2']['Valor'].append(centroid[line][col])
                dic['Capacidade']['Grupo_2']['Ano'].append(ano)
            elif line == 1 and col == 1:
                dic['Dependência_União']['Grupo_2']['Valor'].append(centroid[line][col])
                dic['Dependência_União']['Grupo_2']['Ano'].append(ano)
            elif line == 1 and col == 2:
                dic['Dependência_Estado']['Grupo_2']['Valor'].append(centroid[line][col])
                dic['Dependência_Estado']['Grupo_2']['Ano'].append(ano)
            elif line ==2 and col == 0:
                dic['Capacidade']['Grupo_3']['Valor'].append(centroid[line][col])
                dic['Capacidade']['Grupo_3']['Ano'].append(ano)
            elif line == 2 and col == 1:
                dic['Dependência_União']['Grupo_3']['Valor'].append(centroid[line][col])
                dic['Dependência_União']['Grupo_3']['Ano'].append(ano)
            elif line == 2 and col == 2:
                dic['Dependência_Estado']['Grupo_3']['Valor'].append(centroid[line][col])
                dic['Dependência_Estado']['Grupo_3']['Ano'].append(ano)

    # Criando os Graficos
for indicador in X.columns:
    series = dic[str(indicador)]
    fig, axes = plt.subplots(3, 1)
    axes[0].plot(series['Grupo_1']['Ano'],series['Grupo_1']['Valor'])
    axes[1].plot(series['Grupo_2']['Ano'],series['Grupo_2']['Valor'])
    axes[2].plot(series['Grupo_3']['Ano'],series['Grupo_3']['Valor'])

plt.show()