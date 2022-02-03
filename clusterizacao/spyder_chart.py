import plotly.graph_objects as go 

# Criando função de grafico de teia de aranha 
def web_plot(dataframe):

    # Tratando os dados 
    # Criando categorias 
    categorias = dataframe['Metrica']

    # Removendo as categorias da base 
    data = dataframe.drop(['Metrica'],axis=1)
    
    # Capturando as colunas do dataframe 
    colunas = data.columns 

    # Criando a figura 
    fig = go.Figure()

    # Criando os graficos 
    for columns in colunas:
        fig.add_trace(
            go.Scatterpolar(
                r = data[columns],
                theta = categorias,
                fill = 'toself',
                name = columns
            )
        )
    
    # Atualizando as figuras 
    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
            )
        ),
    showlegend=False
    )

    return fig.show()