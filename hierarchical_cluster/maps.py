# Pacotes 
import geopandas as gdp 
import matplotlib.pyplot as plt
# Criando mapas 
# Importando os arquivos das malhas geograficas 
municipios = '/home/alexandre/Documents/Ciência de Dados/Monografia/Siops_hierarchical_cluster_prediction/hierarchical_cluster/Malhas municipais/23MUE250GC_SIR.shp'

# Criando o grafico 
def maps(dados,codigo_municipio):

    # Importando as malhas geograficas 
    map_df = gdp.read_file(municipios)

    # Transformando os formatos de dados da coluna de codigo 
    map_df["CD_GEOCODM"] = map_df["CD_GEOCODM"].astype('int64')

    # Mergeando os dados 
    map  = map_df.merge(dados,left_on='CD_GEOCODM',right_on=codigo_municipio)

    # Criando o mapa  
    map.plot(column="Cluster")

    return plt.show()