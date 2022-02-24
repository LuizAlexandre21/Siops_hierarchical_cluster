# Pacotes 
from database import *
from data_treatment import * 
#from cluster import *
#from metricas import * 
from spyder_chart import * 
from sklearn.cluster import KMeans

# Importando os dados  
data = data().dataframe()

# Selecionando 20 municipios aleatórios 
# Criando 20 indices aleatórios 
loc = np.random.randint(178,size=(20))

# Selecionando as 20 cidades 
cidades = data['Municipio'].loc[loc].to_list()

# Filtrando a tabela 
data = data[data['Municipio'].str.contains(cidades[0]+"|"+cidades[1]+"|"+cidades[2]+"|"+cidades[3]+"|"+cidades[4]+"|"+cidades[5]+"|"+cidades[6]+"|"+cidades[7]+"|"+cidades[8]+"|"+cidades[9]+"|"+cidades[10]+"|"+cidades[11]+"|"+cidades[12]+"|"+cidades[13]+"|"+cidades[14]+"|"+cidades[15]+"|"+cidades[16]+"|"+cidades[17]+"|"+cidades[18]+"|"+cidades[19])]

# Apresentando os municipios aleatórios 
print(cidades)

# Criando clusterizações 
for ano in range(2013,2020):

    # Filtrando os dados 
    data_filter = data[data['Ano']<=ano]

    # Normalizando os dados 
    data_scaled = normalize(data_filter[['Capacidade','Dependência_União','Dependência_Estado']])
    data_scaled = pd.DataFrame(data_scaled,columns=['Capacidade','Dependência_União','Dependência_Estado'])

    # Criando dendograma  
    #plt.figure(figsize=(10, 7))  
    #plt.title("Dendrograms")  
    #dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
    #plt.savefig("Dendrograms+"+str(ano))

    # Criando clusterização 
    cluster = KMeans(n_clusters=3)  
    cluster.fit_predict(data_scaled)

    fig,axs = plt.subplots(3,1)
    axs[0].scatter(data_scaled["Dependência_União"], data_scaled["Capacidade"], c=cluster.labels_)
    axs[0].legend()
    axs[1].scatter(data_scaled["Dependência_Estado"],data_scaled["Capacidade"],c=cluster.labels_)
    axs[1].legend()
    axs[2].scatter(data_scaled["Dependência_Estado"],data_scaled["Dependência_União"],c=cluster.labels_)
    axs[2].legend()
    plt.savefig("Clusterização_"+str(ano))

    # 