# Pacotes 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import scipy as sc 
from database import * 
from sklearn.metrics import classification_report,confusion_matrix,adjusted_rand_score,rand_score,fowlkes_mallows_score,jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize


# Importando os dados 
municipios = Indicador_Capacidade.select(Indicador_Capacidade.Municipio,Indicador_Capacidade.Estado,Indicador_Capacidade.Codigo,Indicador_Capacidade.Ano,Indicador_Capacidade.Capacidade,Indicador_Dependencia.Dependência_União,Indicador_Dependencia.Dependência_Estado,Indicador_Dependencia_Sus.Dependência_União,Indicador_Dependencia_Sus.Dependência_Estado,Populacao.populacao_estimada,Classificação_Municipios.IDH,Classificação_Municipios.Região,Classificação_Municipios.Porte,Classificação_Municipios.Macroregião,ProdutoInternoBrutoCe.produto_interno_bruto,ProdutoInternoBrutoCe.semiárido).join(Indicador_Dependencia, on=((Indicador_Capacidade.Codigo ==Indicador_Dependencia.Codigo) & (Indicador_Capacidade.Ano == Indicador_Dependencia.Ano))).join(Indicador_Dependencia_Sus, on=((Indicador_Capacidade.Codigo ==Indicador_Dependencia_Sus.Codigo) & (Indicador_Capacidade.Ano == Indicador_Dependencia_Sus.Ano))).join(Populacao, on=((Indicador_Capacidade.Codigo == Populacao.Codigo) & (Indicador_Capacidade.Ano == Populacao.Ano))).join(Classificação_Municipios, on = (Indicador_Capacidade.Municipio == Classificação_Municipios.Municipio)).join(ProdutoInternoBrutoCe, on =((Indicador_Capacidade.Codigo == ProdutoInternoBrutoCe.codigo_municipio) & (Indicador_Capacidade.Ano == ProdutoInternoBrutoCe.ano)))

# Tratando os dados 
data = pd.DataFrame(municipios.dicts())

# Normalizando os dados 
data_scaled = normalize(data[['Capacidade','Dependência_União','Dependência_Estado']])
data_scaled = pd.DataFrame(data_scaled,columns=['Capacidade','Dependência_União','Dependência_Estado'])

# Criando variada observavel e observada
X = data[['Capacidade','Dependência_União','Dependência_Estado']].values
Y_pre = data['IDH']
Y_pre=Y_pre.replace('Medio',1)
Y_pre=Y_pre.replace('Alto',0)
Y_pre=Y_pre.replace('Baixo',2)
Y = Y_pre

# Criando Conjunto de treino e Conjunto de teste 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

# Treinando e Prevendo 
classifier = KNeighborsClassifier(len(np.unique(Y)))
classifier.fit(X_train,y_train)

# Prevendo 
y_pred = classifier.predict(X_test)

# Evalorando o algoritimo 
# Matriz de Confusão
matriz_confusão = confusion_matrix(y_test, y_pred)

# Metricas 
metricas = classification_report(y_test, y_pred)

# Rand_score 
Rand_score = rand_score(y_test, y_pred)

# Rand_score_adjusting 
Adjusted_rand_score = adjusted_rand_score(y_test, y_pred)

# fowlkes_mallows_score
Folkes = fowlkes_mallows_score(y_test,y_pred)

# jaccard_score
Jaccard = jaccard_score(y_test,y_pred,average='macro')

# Visualizando os resultados 
fig,axs = plt.subplots(3,1)
axs[0].scatter(X_test[:,0], X_test[:,1], c=classifier.labels_)
axs[1].scatter(X_test[:,1],X_test[:,2],c=classifier.labels_)
axs[2].scatter(X_test[:,2],X_test[:,0],c=classifier.labels_)
plt.show()