# Pacotes 
from sklearn.cluster import OPTICS
from sklearn.model_selection import GridSearchCV
from data_treatment import *
import plotly.express as px 

# Importandoo os dados 
# Amostras de Treino e Teste  
X_train, X_test = data().main()

# Dataframe 
dados = data().normalized(data().dataframe())[['Capacidade','Dependência_União','Dependência_Estado','IDH']]
dados,rotulate = data.rotulate(1,dados,"IDH")

# Estimando o modelo 
params = {
    'damping':[0.5,0.6,0.7,0.8,0.9]
}

# Criando o modelo --> cross validation 
model = AffinityPropagation() 

# classificando --> cross validation 
classifier = GridSearchCV(model,params,cv=5,scoring='f1')
classifier.fit(X_train) 

# Melhor modelo 
model = classifier.best_estimator_

# Classificando os clusters 
model.fit_predict(dados)

# Criando os graficos 
fig,axs = plt.subplots(3,1)
axs[0].scatter(dados["Dependência_União"], dados["Capacidade"], c=labels_)
axs[1].scatter(dados["Dependência_Estado"],dados["Capacidade"],c=labels_)
axs[2].scatter(dados["Dependência_Estado"],dados["Dependência_União"],c=labels_)
plt.show()
