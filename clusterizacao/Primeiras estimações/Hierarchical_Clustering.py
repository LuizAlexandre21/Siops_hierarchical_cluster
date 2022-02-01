# Pacotes 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from database import *
import scipy.cluster.hierarchy as shc
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score,rand_score,fowlkes_mallows_score,jaccard_score,

# Importando os dados
municipios = Indicador_Capacidade.select(Indicador_Capacidade.Municipio,Indicador_Capacidade.Estado,Indicador_Capacidade.Codigo,Indicador_Capacidade.Ano,Indicador_Capacidade.Capacidade,Indicador_Dependencia.Dependência_União,Indicador_Dependencia.Dependência_Estado,Indicador_Dependencia_Sus.Dependência_União,Indicador_Dependencia_Sus.Dependência_Estado,Populacao.populacao_estimada,Classificação_Municipios.IDH,Classificação_Municipios.Região,Classificação_Municipios.Porte,Classificação_Municipios.Macroregião,ProdutoInternoBrutoCe.produto_interno_bruto,ProdutoInternoBrutoCe.semiárido).join(Indicador_Dependencia, on=((Indicador_Capacidade.Codigo ==Indicador_Dependencia.Codigo) & (Indicador_Capacidade.Ano == Indicador_Dependencia.Ano))).join(Indicador_Dependencia_Sus, on=((Indicador_Capacidade.Codigo ==Indicador_Dependencia_Sus.Codigo) & (Indicador_Capacidade.Ano == Indicador_Dependencia_Sus.Ano))).join(Populacao, on=((Indicador_Capacidade.Codigo == Populacao.Codigo) & (Indicador_Capacidade.Ano == Populacao.Ano))).join(Classificação_Municipios, on = (Indicador_Capacidade.Municipio == Classificação_Municipios.Municipio)).join(ProdutoInternoBrutoCe, on =((Indicador_Capacidade.Codigo == ProdutoInternoBrutoCe.codigo_municipio) & (Indicador_Capacidade.Ano == ProdutoInternoBrutoCe.ano)))

# Tratando os dados 
data = pd.DataFrame(municipios.dicts())

# Normalizando os dados 
data_scaled = normalize(data[['Capacidade','Dependência_União','Dependência_Estado']])
data_scaled = pd.DataFrame(data_scaled,columns=['Capacidade','Dependência_União','Dependência_Estado'])

# Criando dendograma 
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))

# Criando clusterização
for num in range(2,8):
    cluster = AgglomerativeClustering(n_clusters=num, affinity='euclidean', linkage='ward')  
    cluster.fit_predict(data_scaled)

    # Visualizando os resultados 
    fig,axs = plt.subplots(3,1)
    axs[0].scatter(data_scaled["Dependência_União"], data_scaled["Capacidade"], c=cluster.labels_)
    axs[1].scatter(data_scaled["Dependência_Estado"],data_scaled["Capacidade"],c=cluster.labels_)
    axs[2].scatter(data_scaled["Dependência_Estado"],data_scaled["Dependência_União"],c=cluster.labels_)
    plt.show()