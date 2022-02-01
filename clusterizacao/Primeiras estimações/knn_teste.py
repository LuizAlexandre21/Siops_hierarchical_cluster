# Pacotes 
import pandas as pd 
import matplotlib.pyplot as plt 
from database import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np 
import scipy as sc 

# Importando os dados 
municipios = Indicador_Capacidade.select(Indicador_Capacidade.Municipio,Indicador_Capacidade.Estado,Indicador_Capacidade.Codigo,Indicador_Capacidade.Ano,Indicador_Capacidade.Capacidade,Indicador_Dependencia.Dependência_União,Indicador_Dependencia.Dependência_Estado,Indicador_Dependencia_Sus.Dependência_União,Indicador_Dependencia_Sus.Dependência_Estado,Populacao.populacao_estimada,Classificação_Municipios.IDH,Classificação_Municipios.Região,Classificação_Municipios.Porte,Classificação_Municipios.Macroregião,ProdutoInternoBrutoCe.produto_interno_bruto,ProdutoInternoBrutoCe.semiárido).join(Indicador_Dependencia, on=((Indicador_Capacidade.Codigo ==Indicador_Dependencia.Codigo) & (Indicador_Capacidade.Ano == Indicador_Dependencia.Ano))).join(Indicador_Dependencia_Sus, on=((Indicador_Capacidade.Codigo ==Indicador_Dependencia_Sus.Codigo) & (Indicador_Capacidade.Ano == Indicador_Dependencia_Sus.Ano))).join(Populacao, on=((Indicador_Capacidade.Codigo == Populacao.Codigo) & (Indicador_Capacidade.Ano == Populacao.Ano))).join(Classificação_Municipios, on = (Indicador_Capacidade.Municipio == Classificação_Municipios.Municipio)).join(ProdutoInternoBrutoCe, on =((Indicador_Capacidade.Codigo == ProdutoInternoBrutoCe.codigo_municipio) & (Indicador_Capacidade.Ano == ProdutoInternoBrutoCe.ano)))#.where(Indicador_Capacidade.Ano == 2013)
data = pd.DataFrame(municipios.dicts())

# Tratando os dados
# Removendo Outliers 
data=data[data['Municipio']!='Fortaleza']

# Amplitude
amp = (data['produto_interno_bruto'].max()-data['produto_interno_bruto'].min())


# Criando rotulações para o Pib
pib =[]
for prod in data['produto_interno_bruto']:
    if prod <= 2*amp/6:
        pib.append("Muito Baixo")
    elif prod <= 3*amp/6:
        pib.append("Baixo")
    elif prod <= 4*amp/6:
        pib.append("Medio")
    elif prod <= 5*amp/6 :
        pib.append("Alto")
    else: 
        pib.append("Muito Alto")

# Manipulando dados 
data['Pib'] = pib
columns = ['IDH','Região','Macroregião','Porte','Pib']

for number in range(len(columns)):
    fig, axs = plt.subplots(2,2)
    #axs[0] = dendrogram(linkage(base, method = 'ward'))

    # Adicionando os parametros 
    param = columns[number]
    data_filter = data[['Municipio','Codigo','Ano','Capacidade','Dependência_União','Dependência_Estado',param]]

    # Definindo as variaveis observaveis e observada 
    X = data[['Capacidade','Dependência_União','Dependência_Estado']].values
    Y_pre = data[param]

    # Transforamndo rotulos para interiros
    if param == 'IDH':
        Y_pre=Y_pre.replace('Medio',1)
        Y_pre=Y_pre.replace('Alto',0)
        Y_pre=Y_pre.replace('Baixo',2)
        Y = Y_pre
    else:   
        remap = {}
        for i in range(len(np.unique(Y_pre))):
            lista = list(np.unique(Y_pre))
            remap[str(lista[i])]=i
        Y_pre = Y_pre.replace(remap)
        Y = Y_pre        

    # Criando o conjunto de teste e treino 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

    # Treinando e Prevendo 
    # Treinando 
    classifier = AgglomerativeClustering(n_clusters = len(np.unique(Y)), affinity = 'euclidean', linkage = 'ward')
    classifier.fit(X_train,y_train)

    # Prevendo 
    y_pred = classifier.fit_predict(X_test)

    # Evalorando o algoritimo
    matriz_confusão = confusion_matrix(y_test, y_pred)
    metricas = classification_report(y_test, y_pred)

    # Apresentando os graficos
    axs[0,0].scatter(X_test[y_test==0,0],X_test[y_test==0,1])
    axs[0,0].scatter(X_test[y_test==1,0],X_test[y_test==1,1])
    axs[0,0].scatter(X_test[y_test==2,0],X_test[y_test==2,1])
    axs[1,0].scatter(X_test[y_test==0,1],X_test[y_test==0,2])
    axs[1,0].scatter(X_test[y_test==1,1],X_test[y_test==1,2])
    axs[1,0].scatter(X_test[y_test==2,1],X_test[y_test==2,2])
    axs[0,1].scatter(X_test[y_test==0,2],X_test[y_test==0,0])
    axs[0,1].scatter(X_test[y_test==1,2],X_test[y_test==1,0])
    axs[0,1].scatter(X_test[y_test==2,2],X_test[y_test==2,0])

    plt.show()