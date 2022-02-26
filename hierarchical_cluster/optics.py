# Pacotes 
from sklearn.cluster import OPTICS
from sklearn.model_selection import GridSearchCV
from data_treatment import *
import plotly.express as px 

## Importandoo os dados 
# Amostras de Treino e Teste  
X_train, X_test = data().main()

# Dataframe 
dados = data().normalized(data().dataframe())[['Capacidade','Dependência_União','Dependência_Estado','IDH']]
dados,rotulate = data.rotulate(1,dados,"IDH")

# Estimando o modelo 
params ={
    'max_eps': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    'metric':['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'],
    'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
}

# Criando o modelo --> cross validation 
model =  OPTICS()

# classificando --> cross validation 
classifier = GridSearchCV(model,params,cv=5,scoring='f1')
classifier.fit(X_train) 

# Melhor estimador 
model = classifier.best_estimator_

# Classificando os clusters 
model.fit_predict(dados)


# Criando os graficos 
fig,axs = plt.subplots(3,1)
axs[0].scatter(dados["Dependência_União"], dados["Capacidade"], c=labels_)
axs[1].scatter(dados["Dependência_Estado"],dados["Capacidade"],c=labels_)
axs[2].scatter(dados["Dependência_Estado"],dados["Dependência_União"],c=labels_)
plt.show()
