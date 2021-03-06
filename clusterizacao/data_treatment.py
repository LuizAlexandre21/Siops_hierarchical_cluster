# Pacotes 
from database import *
import pandas as pd
import numpy as np
import statistics as st
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

municipios = Indicador_Capacidade.select(Indicador_Capacidade.Municipio,Indicador_Capacidade.Estado,Indicador_Capacidade.Codigo,Indicador_Capacidade.Ano,Indicador_Capacidade.Capacidade,Indicador_Dependencia.Dependência_União,Indicador_Dependencia.Dependência_Estado,Indicador_Dependencia_Sus.Dependência_União,Indicador_Dependencia_Sus.Dependência_Estado,Populacao.populacao_estimada,Classificação_Municipios.IDH,Classificação_Municipios.Região,Classificação_Municipios.Porte,Classificação_Municipios.Macroregião,ProdutoInternoBrutoCe.produto_interno_bruto,ProdutoInternoBrutoCe.semiárido).join(Indicador_Dependencia, on=((Indicador_Capacidade.Codigo ==Indicador_Dependencia.Codigo) & (Indicador_Capacidade.Ano == Indicador_Dependencia.Ano))).join(Indicador_Dependencia_Sus, on=((Indicador_Capacidade.Codigo ==Indicador_Dependencia_Sus.Codigo) & (Indicador_Capacidade.Ano == Indicador_Dependencia_Sus.Ano))).join(Populacao, on=((Indicador_Capacidade.Codigo == Populacao.Codigo) & (Indicador_Capacidade.Ano == Populacao.Ano))).join(Classificação_Municipios, on = (Indicador_Capacidade.Municipio == Classificação_Municipios.Municipio)).join(ProdutoInternoBrutoCe, on =((Indicador_Capacidade.Codigo == ProdutoInternoBrutoCe.codigo_municipio) & (Indicador_Capacidade.Ano == ProdutoInternoBrutoCe.ano)))#.where(Indicador_Capacidade.Ano == 2013)


class data():
    # TODO: Incluir os indicadores de dependência do SUS
    # Importando os dados 
    def dataframe(self):
        return pd.DataFrame(municipios.dicts())
    
    # Normalizando os indicadores 
    def normalized(self,data):
        # TODO: Incluir os indicadores de dependência do SUS
        datas = data
        data_scaled = normalize(data[['Capacidade','Dependência_União','Dependência_Estado']] )
        data_scaled = pd.DataFrame(data_scaled,columns=['Capacidade','Dependência_União','Dependência_Estado'])
        datas['Capacidade'],datas['Dependência_União'],datas['Dependência_Estado'] = data_scaled['Capacidade'],data_scaled['Dependência_União'],data_scaled['Dependência_Estado']
        return datas 

    # Criando rotulos numericos 
    def rotulate(self,data,column):
        relabel = {}
        unico = np.unique(data[column])
        for num in range(len(unico)):
            label = unico[num]
            data[column]=data[column].replace(label,num)
            relabel[label] = num
        return data,relabel 

    # Filtrando o dataframe 
    def filter_data(self,data,x,y):
        X = data[x]
        y = data[y]
        return X,y 

    # Criando a analise  
    def main(self):
        data = self.dataframe()
        data_normalized = self.normalized(data)
        data_rotulate,data_label = self.rotulate(data,'IDH')
        X,y = self.filter_data(data_rotulate,['Capacidade','Dependência_União','Dependência_Estado'],['IDH'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
        return X_train, X_test, y_train, y_test 

