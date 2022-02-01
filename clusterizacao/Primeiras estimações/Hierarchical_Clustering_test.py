# Pacores 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from database import *
from scipy.cluster import hierarchy as shc 
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score,rand_score,fowlkes_mallows_score,jaccard_score,

# Importando os dados 
municipios = Indicador_Capacidade.select(Indicador_Capacidade.Municipio,Indicador_Capacidade.Estado,Indicador_Capacidade.Codigo,Indicador_Capacidade.Ano,Indicador_Capacidade.Capacidade,Indicador_Dependencia.Dependência_União,Indicador_Dependencia.Dependência_Estado,Indicador_Dependencia_Sus.Dependência_União,Indicador_Dependencia_Sus.Dependência_Estado,Populacao.populacao_estimada,Classificação_Municipios.IDH,Classificação_Municipios.Região,Classificação_Municipios.Porte,Classificação_Municipios.Macroregião,ProdutoInternoBrutoCe.produto_interno_bruto,ProdutoInternoBrutoCe.semiárido).join(Indicador_Dependencia, on=((Indicador_Capacidade.Codigo ==Indicador_Dependencia.Codigo) & (Indicador_Capacidade.Ano == Indicador_Dependencia.Ano))).join(Indicador_Dependencia_Sus, on=((Indicador_Capacidade.Codigo ==Indicador_Dependencia_Sus.Codigo) & (Indicador_Capacidade.Ano == Indicador_Dependencia_Sus.Ano))).join(Populacao, on=((Indicador_Capacidade.Codigo == Populacao.Codigo) & (Indicador_Capacidade.Ano == Populacao.Ano))).join(Classificação_Municipios, on = (Indicador_Capacidade.Municipio == Classificação_Municipios.Municipio)).join(ProdutoInternoBrutoCe, on =((Indicador_Capacidade.Codigo == ProdutoInternoBrutoCe.codigo_municipio) & (Indicador_Capacidade.Ano == ProdutoInternoBrutoCe.ano)))
data = pd.DataFrame(municipios.dicts())

# Filtrando os 20 municipios aleatórios 
loc = np.random.randint(178,size=(20))
cidades = data['Municipio'].loc[loc].to_list()
data = data[data['Municipio'].str.contains(cidades[0]+"|"+cidades[1]+"|"+cidades[2]+"|"+cidades[3]+"|"+cidades[4]+"|"+cidades[5]+"|"+cidades[6]+"|"+cidades[7]+"|"+cidades[8]+"|"+cidades[9]+"|"+cidades[10]+"|"+cidades[11]+"|"+cidades[12]+"|"+cidades[13]+"|"+cidades[14]+"|"+cidades[15]+"|"+cidades[16]+"|"+cidades[17]+"|"+cidades[18]+"|"+cidades[19])]

# Obtendo os resultados
for ano in range(2013,2020):
    data_filter = data[data['Ano']<=ano]

    # Normalizando os dados 
    data_scaled = normalize(data_filter[['Capacidade','Dependência_União','Dependência_Estado']])
    data_scaled = pd.DataFrame(data_scaled,columns=['Capacidade','Dependência_União','Dependência_Estado'])

    # Criando dendograma  
    plt.figure(figsize=(10, 7))  
    plt.title("Dendrograms")  
    dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
    plt.savefig("Dendrograms+"+str(ano))

    # Criando clusterização 
    cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')  
    cluster.fit_predict(data_scaled)
    fig,axs = plt.subplots(3,1)
    axs[0].scatter(data_scaled["Dependência_União"], data_scaled["Capacidade"], c=cluster.labels_)
    axs[0].legend()
    axs[1].scatter(data_scaled["Dependência_Estado"],data_scaled["Capacidade"],c=cluster.labels_)
    axs[1].legend()
    axs[2].scatter(data_scaled["Dependência_Estado"],data_scaled["Dependência_União"],c=cluster.labels_)
    axs[2].legend()
    plt.savefig("Clusterização_"+str(ano))

