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
                name = columns,
                #legendgroup={"Metodo":['completeness_score', 'fowlkes_mallows_score','mutual_info_score', 'rand_score']}
            )
        )
    
    # Atualizando as figuras 
    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1])),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))

    return fig.show()