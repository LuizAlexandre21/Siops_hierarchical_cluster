# Bibliotecas
from peewee import *
from playhouse.db_url import connect 

# Estabelecendo conexão 
database = connect("mysql://alexandre:34340012@localhost:3306/Data_saude")

# Criando a classe de conexão mysql 
class MySQLBitField(Field):
    field_type = "bit"
    
    def __init__(self,*_,**__):
        pass 
    
#Criando a classe basica do modelo 
class BaseModel(Model):
    class Meta:
        database = database 
        
# População 
class Populacao(BaseModel):
    UF = TextField()
    codigo_munic = IntegerField(column_name='COD. MUNIC') 
    municipio = TextField()
    populacao_estimada = IntegerField() 
    Ano  = TextField()
    Codigo = TextField() 
    class Meta:
        primary_key = False
        table_name = "Populacao"
        
# Classificação Municipios 
class Classificação_Municipios(BaseModel):
    Municipio = TextField()
    Macroregião =  TextField()
    Região = TextField() 
    IDH = TextField() 
    Porte = TextField() 
    class Meta:
        primary_key = False 
        table_name = "Classificações_Municipios" 
        
# Indicador de Capacidade 
class Indicador_Capacidade(BaseModel):
    Municipio = TextField(column_name= 'municipio')
    Estado = TextField(column_name='estado')
    Codigo = IntegerField(column_name='codigo')
    Ano = IntegerField(column_name='ano')
    Capacidade = FloatField()
    class Meta:
        primary_key = False
        table_name = "Indicador_de_Capacidade_do_Municipio"
        
# Indicador de Dependência 
class Indicador_Dependencia(BaseModel):
    Municipio = TextField(column_name='municipio') 
    Estado = TextField(column_name='estado')
    Codigo = TextField(column_name='codigo')
    Ano = TextField(column_name='ano')
    Dependência_União = FloatField()
    Dependência_Estado = FloatField()
    class Meta:
        primary_key =False
        table_name = "Indicadores_de_Dependência"
        
# Indicador de Dependência Sus 
class Indicador_Dependencia_Sus(BaseModel):
    Municipio = TextField(column_name='municipio') 
    Estado = TextField(column_name='estado')
    Codigo = TextField(column_name='codigo')
    Ano = TextField(column_name='ano')
    Dependência_União = FloatField()
    Dependência_Estado = FloatField()
    class Meta:
        primary_key = False
        table_name = "Indicadores_de_Dependência_SUS"
        
# Produto Interno Bruto 
class ProdutoInternoBrutoCe(BaseModel):
    amazônia__legal = TextField(column_name='Amazônia Legal', null=True)
    ano = BigIntegerField(column_name='Ano', null=True)
    codigo_municipio = CharField(column_name='Codigo_municipio', null=True)
    nome_da__região__rural = TextField(column_name='Nome da Região Rural', null=True)
    nome_da__unidade_da__federação = TextField(column_name='Nome da Unidade da Federação', null=True)
    nome_do__município = TextField(column_name='Nome do Município', null=True)
    produto_interno_bruto = FloatField(column_name='Produto_interno_bruto', null=True)
    semiárido = TextField(column_name='Semiárido', null=True)

    class Meta:
        table_name = 'Produto_interno_bruto_ce'
        primary_key = False