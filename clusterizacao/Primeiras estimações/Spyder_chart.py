import plotly.graph_objects as go
#from metricas import metrics 
import pandas as pd 

# Importando as metricas 
metricas = pd.read_csv("Metricas.csv")

# Categorias 
categories = metricas['Metodo']

# criando a figura 
fig = go.Figure()

for i in metricas.drop(['Unnamed: 0','Metodo'],axis=1).columns :
    fig.add_trace(go.Scatterpolar(
        r=metricas[i],
        theta=categories,
        fill='toself',
        name='Product A'
    ))

fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 5]
    )),
  showlegend=False
)

fig.show()