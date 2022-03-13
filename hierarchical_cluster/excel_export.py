# Pacotes 
import xlsxwriter

# Criando funções de 
def export(dicionario:dict,table:str):

    # Definindo o conjunto de colunas utilizada 
    col = ['A','B','C','D','E','F','G','H','I','J']
    # Definindo a tabela para exportação
    workbook = xlsxwriter.Workbook(table + '.xlsx')
       
    # Criando o formato do titulo da tabela  
    estatitica_format = workbook.add_format({'bold': 1,'border': 1,'align': 'center','valign': 'vcenter','fg_color': 'yellow'}) # Estatisticas 
    free_format = workbook.add_format({'bold': 1,'border': 0,'align': 'right' ,'valign': 'vcenter','fg_color': 'blue'}) # Celulas Livres
    
    # Criando as planilhas 
    for ano in dicionario.keys():

        # Criando a planilha na tabela
        worksheet = workbook.add_worksheet(str(ano))
        
        # Separando por clusters as tabelas 
        for cluster in dicionario[ano].keys():
            
            # Tabela 1 - Municipios             
            # Criando o corpo da tabela
            for count,mun in enumerate(dicionario[ano][cluster]['Municipios']):
                
                # Coluna e linha
                col = 0 
                lin = 3
                print(col,lin)
                # Criando as posições de cada informação
                if count % 5 != 0: 

                    # Adicionando a informação 
                    worksheet.write(lin,col,mun)
                    lin +=1 
                
                else:
                    col +=1
                    lin = 3
                    worksheet.write(lin,col,mun)
                    lin+=1
